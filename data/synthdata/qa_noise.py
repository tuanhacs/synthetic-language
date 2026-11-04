"""Reproducible train-only flips of individual encoded QA codeword occurrences."""
from __future__ import annotations

import hashlib
import heapq
import json
import random
from collections import Counter
from dataclasses import replace
from typing import Sequence

from .config import TrainNoiseConfig
from .qa import QASample

VERSION = "qa-occurrence-bit-flip-v1"


def _identity(sample: QASample, index: int) -> str:
    return sample.sample_id or json.dumps((index, sample.query_vertices, sample.answer_walk), separators=(",", ":"))


def clean_sample(sample: QASample) -> QASample:
    return replace(sample, noised_bits=None, noise_edits=())


def noise_report(samples: Sequence[QASample], cfg: TrainNoiseConfig) -> dict:
    counts = Counter()
    checksum = hashlib.sha256()
    for index, sample in enumerate(samples):
        if sample.noised_bits is None:
            continue
        counts["corrupted_samples"] += 1
        counts["codeword_occurrences"] += len(sample.noise_edits)
        for edit in sample.noise_edits:
            counts["flipped_bits"] += len(edit["bit_indices"])
            counts[f"{edit['scope']}_occurrences"] += 1
        checksum.update(json.dumps((_identity(sample, index), sample.noised_bits, sample.noise_edits),
                                   sort_keys=True, separators=(",", ":")).encode())
    return {
        "version": VERSION, "config": cfg.to_dict(), "train_samples": len(samples),
        **{name: counts[name] for name in ("corrupted_samples", "codeword_occurrences", "flipped_bits",
                                         "query_occurrences", "answer_occurrences")},
        "actual_sample_rate": counts["corrupted_samples"] / len(samples) if samples else 0.0,
        "selection": "lowest SHA256 ranks; floor(train_samples * sample_rate)",
        "cot_segments_corrupted": False, "corruption_sha256": checksum.hexdigest(),
    }


def apply_train_noise(samples: Sequence[QASample], cfg: TrainNoiseConfig) -> tuple[tuple[QASample, ...], dict]:
    if any(s.noised_bits is not None for s in samples):
        raise ValueError("train noise must be applied once to clean QA samples")
    n = int(len(samples) * cfg.sample_rate)
    selected = {i for _, i in heapq.nsmallest(n, (
        (hashlib.sha256(f"{VERSION}:{cfg.seed}:select:{_identity(s, i)}".encode()).digest(), i)
        for i, s in enumerate(samples)))}
    output = []
    for index, sample in enumerate(samples):
        if index not in selected:
            output.append(sample)
            continue
        seed = hashlib.sha256(f"{VERSION}:{cfg.seed}:edit:{_identity(sample, index)}".encode()).digest()
        rng = random.Random(int.from_bytes(seed, "big"))
        candidates = []
        for scope, cuts, offset in (("query", sample.query_cuts, 0),
                                    ("answer", sample.answer_cuts, sample.answer_start)):
            if cfg.scope not in (scope, "both"):
                continue
            start = 0
            for occurrence, end in enumerate(cuts):
                if end <= start:
                    raise ValueError("codeword cuts must be strictly increasing")
                candidates.append((scope, occurrence, offset + start, end - start))
                start = end
        k = min(rng.randint(*cfg.codeword_occurrences), len(candidates))
        chars = list(sample.clean_bits)
        edits = []
        for scope, occurrence, start, length in rng.sample(candidates, k):
            bit_count = min(rng.randint(*cfg.bit_flips_per_occurrence), length)
            positions = tuple(sorted(rng.sample(range(length), bit_count)))
            for position in positions:
                ch = chars[start + position]
                if ch not in "01":
                    raise ValueError("noise occurrence points outside a codeword")
                chars[start + position] = "1" if ch == "0" else "0"
            edits.append({"scope": scope, "codeword_index": occurrence, "bit_indices": positions})
        if not edits:
            raise ValueError("selected QA sample has no eligible codeword occurrences")
        output.append(replace(sample, noised_bits="".join(chars), noise_edits=tuple(edits)))
    result = tuple(output)
    return result, noise_report(result, cfg)
