"""Autoregressive generation.

Sampling starts from ``BOS`` (generation from scratch, ``cut = 0``) or from
``BOS + prefix_bits`` (completion of a valid prefix, ``cut = k``) and stops at EOS
or ``max_len``. Generation pre-fills a per-layer KV cache once, then evaluates
only the newly sampled token at each subsequent step.
"""

from __future__ import annotations

from typing import Sequence

import torch

from model import Model


@torch.no_grad()
def generate(
    model: Model,
    tokenizer,
    n: int,
    temperature: float = 1.0,
    max_len: int | None = None,
    prefix_bits: str | Sequence[str] | None = None,
    device: torch.device | str = "cpu",
    rng: torch.Generator | None = None,
    batch_size: int = 64,
    forbid_bos: bool = True,
) -> list[str]:
    """Sample ``n`` sentences and return them as bit strings.

    ``prefix_bits`` is either a single prefix used for every sample, a sequence of
    ``n`` prefixes, or ``None`` (generate from BOS). The returned strings include
    the prefix. A sample that hits ``max_len`` without emitting EOS is returned as
    generated (it will simply be scored as invalid unless it happens to parse).

    PAD (and BOS, unless ``forbid_bos=False``) are masked out of the distribution:
    they are packing artefacts, never part of a sentence.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (greedy decoding is not part of the protocol)")
    max_len = max_len or model.cfg.context_len

    if prefix_bits is None:
        prefixes = [""] * n
    elif isinstance(prefix_bits, str):
        prefixes = [prefix_bits] * n
    else:
        prefixes = list(prefix_bits)
        if len(prefixes) != n:
            raise ValueError(f"got {len(prefixes)} prefixes for n={n}")

    model.eval()
    out: list[str] = []
    for start in range(0, n, batch_size):
        out.extend(
            _generate_batch(
                model, tokenizer, prefixes[start : start + batch_size],
                temperature, max_len, device, rng, forbid_bos,
            )
        )
    return out


def _generate_batch(
    model: Model,
    tokenizer,
    prefixes: list[str],
    temperature: float,
    max_len: int,
    device: torch.device | str,
    rng: torch.Generator | None,
    forbid_bos: bool,
) -> list[str]:
    """Generate one batch with a KV cache.

    RoPE positions and KV-cache lengths must agree within a dense batch, so
    ragged prefixes are split into same-length groups and restored to their
    original order. Fixed-cut evaluation takes the single-group path.
    """
    tok = tokenizer
    seqs = [tok.encode(p, bos=True, eos=False) for p in prefixes]
    lengths = [len(seq) for seq in seqs]
    longest = max(lengths)
    if longest >= max_len:
        raise ValueError(f"prefix length {longest} leaves no room within max_len {max_len}")

    if len(set(lengths)) > 1:
        groups: dict[int, list[int]] = {}
        for row, length in enumerate(lengths):
            groups.setdefault(length, []).append(row)
        ordered = [""] * len(prefixes)
        for rows in groups.values():
            group_out = _generate_batch(
                model,
                tok,
                [prefixes[row] for row in rows],
                temperature,
                max_len,
                device,
                rng,
                forbid_bos,
            )
            for row, string in zip(rows, group_out):
                ordered[row] = string
        return ordered

    b = len(seqs)
    width = lengths[0]
    ids = torch.tensor(seqs, dtype=torch.long, device=device)
    done = torch.zeros(b, dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in range(b)]
    logits, cache = model.forward_cached(ids)

    for step in range(max_len - width):
        last = logits[:, -1, :].float() / temperature
        last[:, tok.PAD] = float("-inf")
        if forbid_bos:
            last[:, tok.BOS] = float("-inf")
        # Sampling happens on CPU: the vocabulary has 5 entries, so the transfer is
        # free, and a CPU generator keeps the draws identical across devices.
        probs = torch.softmax(last, dim=-1).cpu()
        nxt = torch.multinomial(probs, num_samples=1, generator=rng).squeeze(-1).to(device)
        nxt = torch.where(done, torch.full_like(nxt, tok.PAD), nxt)

        for row in range(b):
            if not done[row] and int(nxt[row]) != tok.EOS:
                generated[row].append(int(nxt[row]))
        done |= nxt == tok.EOS
        if bool(done.all()) or step + 1 == max_len - width:
            break
        # Finished rows append PAD only to keep every layer cache rectangular;
        # their subsequent logits are ignored.
        cache_token = torch.where(done, torch.full_like(nxt, tok.PAD), nxt)
        logits, cache = model.forward_cached(cache_token[:, None], cache)

    return [
        prefixes[row] + "".join(tok.ID_TO_TOKEN[i] for i in generated[row]) for row in range(b)
    ]
