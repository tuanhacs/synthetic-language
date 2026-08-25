"""Pool building, splitting and streaming.

Two data modes:
  * **frozen pool** (default) — build a deduplicated pool of ``pool_tokens``
    bits, split it string-disjointly, freeze it to disk with a config hash.
  * **streaming** — an infinite iterator of fresh samples from a seeded
    generator; nothing is stored. For infinite-data / scaling-law runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from .config import DataConfig
from .language import Language, Sample


@dataclass(frozen=True)
class Splits:
    """The three frozen splits, disjoint at string level."""

    train: tuple[Sample, ...]
    valid: tuple[Sample, ...]
    test: tuple[Sample, ...]

    def items(self) -> list[tuple[str, tuple[Sample, ...]]]:
        return [("train", self.train), ("valid", self.valid), ("test", self.test)]

    def sizes(self) -> dict[str, int]:
        return {name: len(split) for name, split in self.items()}

    def total_bits(self) -> dict[str, int]:
        return {name: sum(len(s.bits) for s in split) for name, split in self.items()}


def build_pool(
    language: Language,
    data_cfg: DataConfig,
    rng: random.Random,
    pool_tokens: int | None = None,
    max_attempts_factor: int = 100,
) -> list[Sample]:
    """Sample until the pool holds ``pool_tokens`` bits, deduplicating on ``bits``.

    Duplicates are dropped (they would leak between splits and distort
    memorisation studies) and do not count towards the budget.
    """
    budget = data_cfg.pool_tokens if pool_tokens is None else pool_tokens
    pool: list[Sample] = []
    seen: set[str] = set()
    total = 0
    attempts = 0
    max_attempts = max(1000, budget * max_attempts_factor)
    while total < budget:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"could not reach {budget} bits after {attempts} draws "
                f"(got {total} bits in {len(pool)} unique sentences); "
                "the language may be too small for this pool size"
            )
        sample = language.sample(None, rng)
        if data_cfg.reverse_walks:
            sample = sample.reverse_walk()
        if sample.bits in seen:
            continue
        seen.add(sample.bits)
        pool.append(sample)
        total += len(sample.bits)
    return pool


def split_pool(
    pool: Sequence[Sample], split: Sequence[float], rng: random.Random | None = None
) -> Splits:
    """Split a pool by ratios, disjoint at string level.

    The pool is already deduplicated, so index-disjoint implies string-disjoint.
    ``rng`` shuffles a copy first; pass ``None`` to keep generation order (which
    is already random, and keeps nested subsets prefix-compatible).
    """
    items = list(pool)
    if rng is not None:
        rng.shuffle(items)
    total_weight = float(sum(split))
    if total_weight <= 0:
        raise ValueError("split weights must have a positive sum")
    n = len(items)
    n_train = int(n * split[0] / total_weight)
    n_valid = int(n * split[1] / total_weight)
    train = tuple(items[:n_train])
    valid = tuple(items[n_train : n_train + n_valid])
    test = tuple(items[n_train + n_valid :])
    return Splits(train=train, valid=valid, test=test)


def nested_subsets(
    train: Sequence[Sample], sizes: Sequence[int], unit: str = "tokens"
) -> dict[int, tuple[Sample, ...]]:
    """Prefix subsets of the training split, so smaller sets nest inside larger.

    ``unit='tokens'`` cuts at cumulative bit counts, ``unit='sentences'`` at
    sentence counts. Requested sizes larger than the split are clipped.
    """
    if unit not in ("tokens", "sentences"):
        raise ValueError("unit must be 'tokens' or 'sentences'")
    out: dict[int, tuple[Sample, ...]] = {}
    for size in sorted(sizes):
        if unit == "sentences":
            out[size] = tuple(train[:size])
            continue
        total, cut = 0, 0
        for i, sample in enumerate(train):
            if total >= size:
                break
            total += len(sample.bits)
            cut = i + 1
        out[size] = tuple(train[:cut])
    return out


def stream(language: Language, data_cfg: DataConfig, seed: int | None = None) -> Iterator[Sample]:
    """Infinite iterator of fresh samples (streaming mode, nothing stored)."""
    rng = random.Random(data_cfg.seed if seed is None else seed)
    while True:
        sample = language.sample(None, rng)
        yield sample.reverse_walk() if data_cfg.reverse_walks else sample


def pool_stats(pool: Sequence[Sample]) -> dict[str, Any]:
    """Basic statistics of a pool / split, stored in the manifest."""
    if not pool:
        return {"num_sentences": 0, "total_bits": 0}
    lengths = [len(s.bits) for s in pool]
    walks = [len(s.walk) for s in pool]
    words = [end - start for s in pool for start, end in zip((0,) + s.cuts[:-1], s.cuts)]
    return {
        "num_sentences": len(pool),
        "total_bits": sum(lengths),
        "sentence_bits": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "walk_len": {"min": min(walks), "max": max(walks), "mean": sum(walks) / len(walks)},
        "codeword_bits": {
            "min": min(words),
            "max": max(words),
            "mean": sum(words) / len(words),
        },
        "num_noised": sum(1 for s in pool if s.noised_bits is not None),
    }
