"""Noise transforms applied *after* the sampler (hooks for later experiments).

``gamma`` is the fraction of samples that get corrupted at all; ``rho`` is the
per-unit corruption rate inside a corrupted sample. The clean bits, walk and
cuts are always preserved in the record — the corrupted string lands in
``noised_bits``, so every analysis can compare against the ground truth.

Types:
  * ``bit-flip``    — surface noise, keeps length and the segmentation frame.
  * ``bit-delete``  — drops bits, breaks the frame and shortens the sentence.
  * ``vertex-noise``— resamples walk vertices *before* re-encoding: a clean
    encoding of an (almost surely) invalid walk, i.e. semantic-level noise.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from .config import NoiseConfig
from .language import Language, Sample


def bit_flip(sample: Sample, rho: float, rng: random.Random) -> Sample:
    """Flip every bit independently with probability ``rho``."""
    bits = "".join(
        ("1" if ch == "0" else "0") if rng.random() < rho else ch for ch in sample.bits
    )
    return Sample(sample.bits, sample.walk, sample.cuts, noised_bits=bits)


def bit_delete(sample: Sample, rho: float, rng: random.Random) -> Sample:
    """Delete every bit independently with probability ``rho`` (never all of them)."""
    kept = [ch for ch in sample.bits if rng.random() >= rho]
    if not kept:  # keep at least one bit so the record stays a string
        kept = [sample.bits[rng.randrange(len(sample.bits))]]
    return Sample(sample.bits, sample.walk, sample.cuts, noised_bits="".join(kept))


def vertex_noise(
    sample: Sample, language: Language, rho: float, rng: random.Random
) -> Sample:
    """Resample a fraction ``rho`` of the walk's vertices, then re-encode cleanly."""
    walk = list(sample.walk)
    for i in range(len(walk)):
        if rng.random() < rho:
            walk[i] = rng.randrange(language.graph.num_vertices)
    corrupted = language.encode(walk, rng)
    return Sample(sample.bits, sample.walk, sample.cuts, noised_bits=corrupted.bits)


def apply_noise(
    pool: Sequence[Sample],
    noise_cfg: NoiseConfig | None,
    rng: random.Random,
    language: Language | None = None,
) -> list[Sample]:
    """Corrupt a fraction ``gamma`` of the pool; the rest keeps ``noised_bits=None``.

    ``vertex-noise`` needs the ``language`` to re-encode the corrupted walk.
    """
    if noise_cfg is None:
        return list(pool)
    if noise_cfg.type == "vertex-noise" and language is None:
        raise ValueError("vertex-noise requires the language to re-encode walks")

    out: list[Sample] = []
    for sample in pool:
        if rng.random() >= noise_cfg.gamma:
            out.append(sample)
            continue
        if noise_cfg.type == "bit-flip":
            out.append(bit_flip(sample, noise_cfg.rho, rng))
        elif noise_cfg.type == "bit-delete":
            out.append(bit_delete(sample, noise_cfg.rho, rng))
        elif noise_cfg.type == "vertex-noise":
            assert language is not None
            out.append(vertex_noise(sample, language, noise_cfg.rho, rng))
        else:
            raise ValueError(f"unknown noise type {noise_cfg.type!r}")
    return out


def noised_fraction(pool: Iterable[Sample]) -> float:
    """Share of records that actually carry a corrupted string."""
    total = corrupted = 0
    for sample in pool:
        total += 1
        corrupted += sample.noised_bits is not None
    return corrupted / total if total else 0.0
