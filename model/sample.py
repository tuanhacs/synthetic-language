"""Autoregressive generation.

Sampling starts from ``BOS`` (generation from scratch, ``cut = 0``) or from
``BOS + prefix_bits`` (completion of a valid prefix, ``cut = k``) and stops at EOS
or ``max_len``. Batched, with a plain KV-free forward pass: contexts are short
(<= 512 tokens) and the models are tiny, so recomputing the prefix each step is
fast enough and keeps the model code free of cache plumbing.
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
    """Generate one batch; prefixes of unequal length are left-padded logically.

    Ragged prefixes are handled by grouping: sequences are grown together and a
    finished row simply stops being updated. Prefixes are usually all the same
    length (a fixed cut), so the common case is a single dense batch.
    """
    tok = tokenizer
    seqs = [tok.encode(p, bos=True, eos=False) for p in prefixes]
    max_prefix = max(len(s) for s in seqs)
    if max_prefix >= max_len:
        raise ValueError(f"prefix length {max_prefix} leaves no room within max_len {max_len}")

    b = len(seqs)
    # Right-pad to a rectangle and track true lengths, so short rows keep growing
    # from their own position.
    width = max_prefix
    ids = torch.full((b, width), tok.PAD, dtype=torch.long, device=device)
    for row, seq in enumerate(seqs):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long, device=device)
    done = torch.zeros(b, dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in range(b)]

    for _ in range(max_len - width):
        logits = model(ids)  # (B, T, V)
        last = logits[torch.arange(b, device=device), lengths - 1, :].float() / temperature
        last[:, tok.PAD] = float("-inf")
        if forbid_bos:
            last[:, tok.BOS] = float("-inf")
        # Sampling happens on CPU: the vocabulary has 5 entries, so the transfer is
        # free, and a CPU generator keeps the draws identical across devices.
        probs = torch.softmax(last, dim=-1).cpu()
        nxt = torch.multinomial(probs, num_samples=1, generator=rng).squeeze(-1).to(device)
        nxt = torch.where(done, torch.full_like(nxt, tok.PAD), nxt)

        ids = torch.cat([ids, torch.full((b, 1), tok.PAD, dtype=torch.long, device=device)], dim=1)
        alive = ~done
        ids[alive, lengths[alive]] = nxt[alive]
        for row in range(b):
            if not done[row] and int(nxt[row]) != tok.EOS:
                generated[row].append(int(nxt[row]))
        done |= nxt == tok.EOS
        lengths = lengths + alive.long()
        if bool(done.all()):
            break

    return [
        prefixes[row] + "".join(tok.ID_TO_TOKEN[i] for i in generated[row]) for row in range(b)
    ]
