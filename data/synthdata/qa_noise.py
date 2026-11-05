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
QUERY_NOISE_VERSION = "qa-test-query-bit-flip-v1"
RECOVERABLE_VERSION = "qa-recoverable-query-one-bit-v1"


class QueryRecoveryChecker:
    """Recognize all private-word queries within Hamming radius one.

    Trie/DP explores unknown word boundaries and all configured query lengths.
    The divergence flag avoids enumerating exponentially many vertex sequences.
    No answer, graph adjacency, original cuts or split labels enter recognition.
    """

    def __init__(self, language, query_len: tuple[int, int]):
        self.query_len = tuple(query_len)
        if not 1 <= self.query_len[0] <= self.query_len[1]:
            raise ValueError("invalid recovery query_len")
        owners = {}
        for vertex, book in enumerate(language.codebooks.books):
            for word in book:
                owners.setdefault(word, set()).add(vertex)
        self.all_words = set(owners)
        self.children = [{}]
        self.terminals = [None]
        for word, vertices in owners.items():
            if len(vertices) != 1:
                continue  # QA prompts use private words, just like the encoder.
            node = 0
            for bit in word:
                if bit not in self.children[node]:
                    self.children[node][bit] = len(self.children)
                    self.children.append({})
                    self.terminals.append(None)
                node = self.children[node][bit]
            self.terminals[node] = next(iter(vertices))

    def uniquely_recovers(self, bits: str, expected: tuple[int, ...]) -> bool:
        # State: consumed bits, trie node, flips used, completed words, diverged.
        pending = [(0, 0, 0, 0, False)]
        visited = set()
        found_original = False
        lo, hi = self.query_len
        while pending:
            state = pending.pop()
            if state in visited:
                continue
            visited.add(state)
            pos, node, used, count, different = state
            vertex = self.terminals[node]
            if vertex is not None and count < hi:
                next_count = count + 1
                diverged = different or count >= len(expected) or vertex != expected[count]
                if pos == len(bits) and lo <= next_count <= hi:
                    if diverged or next_count != len(expected):
                        return False
                    found_original = True
                elif pos < len(bits) and next_count < hi:
                    pending.append((pos, 0, used, next_count, diverged))
            if pos < len(bits):
                for bit, child in self.children[node].items():
                    cost = used + (bit != bits[pos])
                    if cost <= 1:
                        pending.append((pos + 1, child, cost, count, different))
        return found_original


def _apply_recoverable_noise(samples, cfg, *, language, query_len, test=False):
    if language is None or query_len is None:
        raise ValueError("recoverable noise requires language and configured query_len")
    if any(s.noised_bits is not None for s in samples):
        raise ValueError("noise must be applied once to clean QA samples")
    checker = QueryRecoveryChecker(language, query_len)
    namespace = RECOVERABLE_VERSION + (":test" if test else ":train")
    requested = int(len(samples) * cfg.sample_rate)
    selected = {i for _, i in heapq.nsmallest(requested, (
        (hashlib.sha256(f"{namespace}:{cfg.seed}:select:{_identity(s, i)}".encode()).digest(), i)
        for i, s in enumerate(samples)))}
    output = []
    attempts = rejected_word = rejected_ambiguous = skipped = 0
    for index, sample in enumerate(samples):
        if index not in selected:
            output.append(sample)
            continue
        rng = random.Random(int.from_bytes(hashlib.sha256(
            f"{namespace}:{cfg.seed}:edit:{_identity(sample, index)}".encode()).digest(), "big"))
        positions = list(range(len(sample.query_bits)))
        rng.shuffle(positions)
        accepted = None
        for position in positions:
            attempts += 1
            occurrence = next(i for i, end in enumerate(sample.query_cuts) if position < end)
            start = sample.query_cuts[occurrence - 1] if occurrence else 0
            end = sample.query_cuts[occurrence]
            bits = sample.query_bits
            noisy = bits[:position] + ("1" if bits[position] == "0" else "0") + bits[position + 1:]
            if noisy[start:end] in checker.all_words:
                rejected_word += 1
                continue
            if not checker.uniquely_recovers(noisy, sample.query_vertices):
                rejected_ambiguous += 1
                continue
            accepted = replace(sample, noised_bits=noisy + sample.clean_bits[len(bits):],
                noise_edits=({"scope": "query", "codeword_index": occurrence,
                              "bit_indices": (position - start,)},))
            break
        if accepted is None:
            skipped += 1
        output.append(accepted if accepted is not None else sample)
    result = tuple(output)
    report = noise_report(result, cfg)
    report.update(version=RECOVERABLE_VERSION, requested_samples=requested,
                  skipped_no_safe_flip=skipped, candidate_attempts=attempts,
                  rejected_valid_codeword=rejected_word, rejected_ambiguous_query=rejected_ambiguous,
                  recovery_radius_bits=1, query_len=list(query_len),
                  recovery_dictionary="private query codewords; all segmentations and vertex sequences",
                  selection="SHA256-ranked requested subset; unsafe samples left clean, no backfill")
    if test:
        report["test_samples"] = report.pop("train_samples")
    return result, report


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
        "version": RECOVERABLE_VERSION if cfg.mode == "recoverable" else VERSION,
        "config": cfg.to_dict(), "train_samples": len(samples),
        **{name: counts[name] for name in ("corrupted_samples", "codeword_occurrences", "flipped_bits",
                                         "query_occurrences", "answer_occurrences")},
        "actual_sample_rate": counts["corrupted_samples"] / len(samples) if samples else 0.0,
        "selection": "lowest SHA256 ranks; floor(train_samples * sample_rate)",
        "cot_segments_corrupted": False, "corruption_sha256": checksum.hexdigest(),
    }


def _bounded_count(bounds: tuple[int, int], available: int, rng: random.Random) -> int:
    if available < 1:
        raise ValueError("no eligible occurrences/bits for query noise")
    return rng.randint(min(bounds[0], available), min(bounds[1], available))


def _apply_noise(samples: Sequence[QASample], cfg: TrainNoiseConfig, *,
                 namespace: str = VERSION, bounded_draw: bool = False) -> tuple[tuple[QASample, ...], dict]:
    if any(s.noised_bits is not None for s in samples):
        raise ValueError("train noise must be applied once to clean QA samples")
    n = int(len(samples) * cfg.sample_rate)
    selected = {i for _, i in heapq.nsmallest(n, (
        (hashlib.sha256(f"{namespace}:{cfg.seed}:select:{_identity(s, i)}".encode()).digest(), i)
        for i, s in enumerate(samples)))}
    output = []
    for index, sample in enumerate(samples):
        if index not in selected:
            output.append(sample)
            continue
        seed = hashlib.sha256(f"{namespace}:{cfg.seed}:edit:{_identity(sample, index)}".encode()).digest()
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
        k = (_bounded_count(cfg.codeword_occurrences, len(candidates), rng) if bounded_draw
             else min(rng.randint(*cfg.codeword_occurrences), len(candidates)))
        chars = list(sample.clean_bits)
        edits = []
        for scope, occurrence, start, length in rng.sample(candidates, k):
            bit_count = (_bounded_count(cfg.bit_flips_per_occurrence, length, rng) if bounded_draw
                         else min(rng.randint(*cfg.bit_flips_per_occurrence), length))
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


def apply_train_noise(samples: Sequence[QASample], cfg: TrainNoiseConfig, *, language=None,
                      query_len=None) -> tuple[tuple[QASample, ...], dict]:
    if cfg.mode == "recoverable":
        return _apply_recoverable_noise(samples, cfg, language=language, query_len=query_len)
    # Keep v1 train-noise draws unchanged for existing frozen datasets.
    return _apply_noise(samples, cfg)


def apply_query_noise(samples: Sequence[QASample], cfg: TrainNoiseConfig, *, language=None,
                      query_len=None) -> tuple[tuple[QASample, ...], dict]:
    """Noise only test prompts, retaining clean vertices/answer reference fields."""
    if cfg.scope != "query":
        raise ValueError("test-query noise must have scope=query")
    if cfg.mode == "recoverable":
        return _apply_recoverable_noise(samples, cfg, language=language, query_len=query_len, test=True)
    result, report = _apply_noise(samples, cfg, namespace=QUERY_NOISE_VERSION, bounded_draw=True)
    report["version"] = QUERY_NOISE_VERSION
    report["test_samples"] = report.pop("train_samples")
    return result, report
