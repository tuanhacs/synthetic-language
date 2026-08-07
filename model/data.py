"""Dataset loading and batching, on top of ``synthdata``.

Two modes, mirroring the data side:

* **frozen** — load a dataset directory with :func:`synthdata.storage.load_dataset`,
  tokenize with ``BitTokenizer``, pack the ``BOS + bits + EOS`` streams into fixed
  windows (``synthdata``'s own :meth:`BitTokenizer.pack`) and sample random batches.
* **streaming** — wrap :func:`synthdata.dataset.stream` into an infinite batch
  iterator, for later infinite-data scaling runs.

Windows are ``context_len + 1`` tokens long, so a batch yields inputs ``w[:, :-1]``
and targets ``w[:, 1:]`` of exactly ``context_len`` positions.

**Loss masking.** Targets equal to PAD (never present with ``drop_last``) or BOS are
excluded from the loss. Sentences are concatenated as in the CFG paper, so no masking
happens across sentence boundaries; BOS is dropped only because it is fully
predictable (it always follows EOS) and would otherwise bias the reported
bits/token *below* the manifest's entropy floor, which is averaged over
``bits + EOS`` tokens only. Excluding it makes the reported loss directly
comparable to the floor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch

import _paths  # noqa: F401  (sys.path shim for synthdata; must precede synthdata imports)
from synthdata.dataset import stream
from synthdata.language import Language, Sample
from synthdata.storage import load_dataset
from synthdata.tokenizer import BitTokenizer

#: Target token ids that never contribute to the loss.
IGNORE_TARGETS = (BitTokenizer.PAD, BitTokenizer.BOS)
IGNORE_INDEX = -100


def pack_windows(
    samples: Sequence[Sample | str], context_len: int, tokenizer: BitTokenizer | None = None
) -> torch.Tensor:
    """Pack sentences into a ``(n_windows, context_len + 1)`` int64 tensor."""
    tok = tokenizer or BitTokenizer()
    windows = list(tok.pack(samples, context_len=context_len + 1, drop_last=True))
    if not windows:
        raise ValueError(
            f"not enough data to fill one window of {context_len + 1} tokens "
            f"({len(samples)} sentences)"
        )
    return torch.tensor(windows, dtype=torch.long)


def make_targets(windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(inputs, targets)`` from packed windows, with ignored targets masked out."""
    inputs, targets = windows[:, :-1], windows[:, 1:].clone()
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for token in IGNORE_TARGETS:
        mask |= targets == token
    targets[mask] = IGNORE_INDEX
    return inputs, targets


@dataclass
class PackedData:
    """A frozen dataset, packed and ready to batch."""

    dataset_dir: Path
    language: Language
    tokenizer: BitTokenizer
    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor
    train_samples: tuple[Sample, ...]
    valid_samples: tuple[Sample, ...]
    test_samples: tuple[Sample, ...]
    manifest: dict
    context_len: int

    @property
    def entropy_floor(self) -> float | None:
        """Bayes-optimal cross-entropy in bits/token, from the dataset manifest."""
        value = self.manifest.get("stats", {}).get("entropy_floor_bits_per_token")
        return None if value is None else float(value)

    def split(self, name: str) -> torch.Tensor:
        return {"train": self.train, "valid": self.valid, "test": self.test}[name]

    def samples(self, name: str) -> tuple[Sample, ...]:
        return {
            "train": self.train_samples,
            "valid": self.valid_samples,
            "test": self.test_samples,
        }[name]

    def token_counts(self) -> dict[str, int]:
        return {
            name: int(self.split(name).numel()) if self.split(name).numel() else 0
            for name in ("train", "valid", "test")
        }


def load_frozen(dataset_dir: str | Path, context_len: int) -> PackedData:
    """Load a synthdata dataset directory and pack all three splits."""
    path = Path(dataset_dir)
    ds = load_dataset(path)
    tok = BitTokenizer()

    def pack_or_empty(samples: tuple[Sample, ...]) -> torch.Tensor:
        try:
            return pack_windows(samples, context_len, tok)
        except ValueError:
            return torch.zeros((0, context_len + 1), dtype=torch.long)

    return PackedData(
        dataset_dir=path,
        language=ds.language,
        tokenizer=tok,
        train=pack_windows(ds.splits.train, context_len, tok),
        valid=pack_or_empty(ds.splits.valid),
        test=pack_or_empty(ds.splits.test),
        train_samples=ds.splits.train,
        valid_samples=ds.splits.valid,
        test_samples=ds.splits.test,
        manifest=ds.manifest,
        context_len=context_len,
    )


class BatchSampler:
    """Random-batch sampler over packed windows, with its own seeded generator."""

    def __init__(self, windows: torch.Tensor, batch_size: int, seed: int = 0) -> None:
        if windows.numel() == 0:
            raise ValueError("no windows to sample from")
        self.windows = windows
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.windows.shape[0]

    def batch(self, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(
            0, self.windows.shape[0], (self.batch_size,), generator=self.generator
        )
        inputs, targets = make_targets(self.windows[idx])
        return inputs.to(device), targets.to(device)

    def sequential_batches(
        self, n_batches: int, device: torch.device | str = "cpu"
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Deterministic pass over the first ``n_batches`` batches (for validation)."""
        total = self.windows.shape[0]
        for start in range(0, min(n_batches * self.batch_size, total), self.batch_size):
            inputs, targets = make_targets(self.windows[start : start + self.batch_size])
            yield inputs.to(device), targets.to(device)


def stream_batches(
    language: Language,
    data_cfg,
    context_len: int,
    batch_size: int,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Infinite batch iterator over freshly sampled sentences (streaming mode).

    ``data_cfg`` is a ``synthdata.config.DataConfig`` (it supplies the walk-length
    range). Sentences are packed exactly as in frozen mode, so the two modes differ
    only in whether the data is stored.
    """
    tok = BitTokenizer()
    samples = stream(language, data_cfg, seed=seed)
    packer = tok.pack(samples, context_len=context_len + 1, drop_last=True)
    while True:
        windows = [next(packer) for _ in range(batch_size)]
        inputs, targets = make_targets(torch.tensor(windows, dtype=torch.long))
        yield inputs.to(device), targets.to(device)


def sentence_batches(
    samples: Sequence[Sample | str],
    batch_size: int,
    tokenizer: BitTokenizer | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Sentence-aligned batches (``BOS + bits + EOS``, PAD-filled) for exact scoring.

    Unlike packed windows, every sentence is scored whole and never split across a
    context boundary, so the resulting bits/token is exactly the quantity the
    entropy floor is defined over.
    """
    tok = tokenizer or BitTokenizer()
    items = [item if isinstance(item, str) else item.bits for item in samples]
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        ids = [tok.encode(bits) for bits in chunk]
        width = max(len(seq) for seq in ids)
        padded = torch.full((len(ids), width), tok.PAD, dtype=torch.long)
        for row, seq in enumerate(ids):
            padded[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        yield make_targets(padded)


def resolve_device(requested: str | None = None) -> torch.device:
    """Device auto-detection: cuda > mps > cpu."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed python + torch RNGs.

    MPS and CUDA kernels may still be nondeterministic; that is acceptable here
    (the *data* is bit-exactly reproducible, the optimisation is not).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
