"""Reusable, codebook-independent QA streams and semantic coverage reports."""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .config import TaskConfig
from .dataset import Splits
from .graphs import GridGraph
from .language import Language
from .qa import (
    LatentQASample, cross_region_direction, cross_region_layout, encode_qa,
    sample_latent_qa,
)

VERSION = 1


def stable_seed(*parts: object) -> int:
    """Unlike Python hash(), stable across processes and PYTHONHASHSEED values."""
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:16], "big")


def semantic_config(graph: GridGraph, task: TaskConfig) -> dict[str, Any]:
    fields = task.to_dict()
    fields.pop("cot_mode", None)
    return {"graph": graph.to_dict(), "task": fields}


def sample_key(sample: LatentQASample) -> str:
    # A query/final-walk pair occurs only once, even if different segment
    # boundaries could explain it. This also avoids duplicate no-CoT strings.
    raw = json.dumps((sample.query_vertices, sample.answer_walk), separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _role(task: TaskConfig, split: str, accepted: int) -> str | None:
    if task.split_mode != "cross-region":
        return None
    roles = ("top-to-bottom", "bottom-to-top") if split == "test" else ("upper", "lower")
    return roles[accepted % 2]


def latent_stream(
    graph: GridGraph, task: TaskConfig, seed: int, split: str,
    seen: set[str],
) -> Iterable[tuple[str, LatentQASample]]:
    """Budget-independent rejection stream. seen includes reserved eval samples."""
    rng = random.Random(stable_seed(VERSION, seed, split))
    accepted = 0
    failures = 0
    while True:
        sample = sample_latent_qa(
            graph, task, rng,
            require_held_out=(split == "test") if task.split_mode == "held-out-pairs" else None,
            cross_role=_role(task, split, accepted),
        )
        key = sample_key(sample) if sample is not None else ""
        if sample is None or key in seen:
            failures += 1
            if failures >= 100_000:
                raise RuntimeError(f"cannot produce a new latent {split} sample; check constraints")
            continue
        failures = 0
        seen.add(key)
        accepted += 1
        yield key, sample


def eligible_pairs(graph: GridGraph, task: TaskConfig, split: str) -> set[tuple[int, int]]:
    pairs = {(u, v) for u in graph.vertices for v in graph.vertices if u != v}
    if task.split_mode == "held-out-pairs" and split != "test":
        held = set(task.held_out_pairs)
        if task.pair_direction == "unordered":
            held |= {(v, u) for u, v in held}
        pairs -= held
    if task.split_mode == "cross-region":
        layout = cross_region_layout(graph.n, task.region_overlap_rows)
        if split == "test":
            outer = layout.top_only | layout.bottom_only
            pairs = {(u, v) for u, v in pairs if u in outer and v in outer}
            if task.query_len[1] == 2:
                pairs = {(u, v) for u, v in pairs if
                         (u in layout.top_only) != (v in layout.top_only)}
        else:
            pairs = {(u, v) for u, v in pairs if
                     any(u in region and v in region for region in (layout.upper, layout.lower))}
    # Endpoints require a simple path within the configured length range.
    # Below the grid diameter, some pairs may be structurally impossible; expose
    # the endpoint-eligible denominator explicitly rather than claim reachability.
    return pairs


def _coverage(counts: Counter, eligible: set) -> dict[str, Any]:
    frequencies = [counts[key] for key in sorted(eligible)]
    unseen = [key for key in sorted(eligible) if not counts[key]]
    return {
        "eligible": len(eligible),
        "seen": len(eligible) - len(unseen),
        "coverage_pct": 100 * (len(eligible) - len(unseen)) / len(eligible) if eligible else None,
        "unseen": [list(key) for key in unseen],
        "frequency_histogram_including_zero": dict(sorted(Counter(frequencies).items())),
        "frequencies": {f"{u},{v}": counts[u, v] for u, v in sorted(eligible)},
        "observed_outside_eligible": [list(key) for key in sorted(set(counts) - eligible)],
    }


class SemanticStats:
    """Streaming statistics; graph edges are counted once on final_walk."""

    def __init__(self, graph: GridGraph, task: TaskConfig, split: str):
        self.graph, self.task, self.split = graph, task, split
        self.n = 0
        self.query_lengths: Counter = Counter()
        self.segment_lengths: Counter = Counter()
        self.answer_lengths: Counter = Counter()
        self.vertices: Counter = Counter()
        self.starts: Counter = Counter()
        self.ends: Counter = Counter()
        self.pairs: Counter = Counter()
        self.edges: Counter = Counter()
        self.queries: set[tuple[int, ...]] = set()
        self.directions: Counter = Counter()
        self.cross_counts: Counter = Counter()
        self.regions: Counter = Counter()

    def update(self, sample: LatentQASample) -> None:
        self.n += 1
        q, walk = sample.query_vertices, sample.answer_walk
        self.queries.add(q)
        self.query_lengths[len(q)] += 1
        self.answer_lengths[len(walk)] += 1
        previous = 0
        for cut in sample.segment_cuts:
            self.segment_lengths[cut - previous + (previous != 0)] += 1
            previous = cut
        self.vertices.update(q)
        self.starts[q[0]] += 1
        self.ends[q[-1]] += 1
        self.pairs.update(zip(q, q[1:]))
        self.edges.update(zip(walk, walk[1:]))
        if self.task.split_mode == "cross-region":
            layout = cross_region_layout(self.graph.n, self.task.region_overlap_rows)
            direction = cross_region_direction(q, self.graph.n, self.task.region_overlap_rows)
            if direction:
                self.directions[direction] += 1
            if self.split != "test":
                # A query wholly in the overlap can fit both regions.
                self.regions["upper" if (self.n - 1) % 2 == 0 else "lower"] += 1
            self.cross_counts[sum(
                (u in layout.top_only and v in layout.bottom_only) or
                (u in layout.bottom_only and v in layout.top_only)
                for u, v in zip(q, q[1:])
            )] += 1

    def report(self) -> dict[str, Any]:
        def histogram(counter: Counter) -> dict:
            return dict(sorted(counter.items()))

        def undirected(counter: Counter) -> Counter:
            result: Counter = Counter()
            for (u, v), count in counter.items():
                result[tuple(sorted((u, v)))] += count
            return result

        pairs = eligible_pairs(self.graph, self.task, self.split)
        edges = set(self.graph.directed_edges())
        if self.task.split_mode == "cross-region" and self.split != "test":
            layout = cross_region_layout(self.graph.n, self.task.region_overlap_rows)
            edges = {(u, v) for u, v in edges if any(
                u in region and v in region for region in (layout.upper, layout.lower)
            )}
        result = {
            "num_samples": self.n,
            "query_pair_occurrences": sum(self.pairs.values()),
            "unique_queries": len(self.queries),
            "histograms": {
                "query_vertices": histogram(self.query_lengths),
                "segment_vertices": histogram(self.segment_lengths),
                "final_walk_vertices": histogram(self.answer_lengths),
                "direction": histogram(self.directions),
                "cross_segment_count": histogram(self.cross_counts),
                "sampled_region": histogram(self.regions),
            },
            "query_vertex_frequency": {v: self.vertices[v] for v in self.graph.vertices},
            "start_vertex_frequency": {v: self.starts[v] for v in self.graph.vertices},
            "end_vertex_frequency": {v: self.ends[v] for v in self.graph.vertices},
            "query_pairs_directed": _coverage(self.pairs, pairs),
            "query_pairs_undirected": _coverage(undirected(self.pairs), {tuple(sorted(p)) for p in pairs}),
            "graph_edges_directed": _coverage(self.edges, edges),
            "graph_edges_undirected": _coverage(undirected(self.edges), {tuple(sorted(p)) for p in edges}),
            "pair_denominator": "endpoint pairs allowed by split; not a simple-path feasibility proof",
        }
        if self.task.split_mode == "held-out-pairs":
            held = set(self.task.held_out_pairs)
            if self.task.pair_direction == "unordered":
                held |= {(v, u) for u, v in held}
            result["held_out_pairs_directed"] = _coverage(
                Counter({p: n for p, n in self.pairs.items() if p in held}), held
            )
        if self.task.split_mode == "cross-region":
            layout = cross_region_layout(self.graph.n, self.task.region_overlap_rows)
            for name, region in (("upper", layout.upper), ("lower", layout.lower)):
                region_pairs = {(u, v) for u, v in pairs if u in region and v in region}
                result[f"{name}_query_pairs_directed"] = _coverage(
                    Counter({p: n for p, n in self.pairs.items() if p in region_pairs}), region_pairs
                )
            cross = {(u, v) for u in layout.top_only for v in layout.bottom_only}
            cross |= {(v, u) for u, v in cross}
            result["cross_region_query_pairs_directed"] = _coverage(
                Counter({p: n for p, n in self.pairs.items() if p in cross}), cross
            )
        return result


def semantic_stats(samples: Iterable, graph: GridGraph, task: TaskConfig, split: str) -> dict:
    stats = SemanticStats(graph, task, split)
    for sample in samples:
        stats.update(sample)
    return stats.report()


def create_latent_pool(
    out: str | Path, graph: GridGraph, task: TaskConfig, seed: int,
    train_pairs: int, valid_pairs: int, test_pairs: int,
    checkpoints: Iterable[int] = (),
) -> dict:
    if task.type != "path-qa" or min(train_pairs, valid_pairs, test_pairs) < 1:
        raise ValueError("latent pools require path-qa and positive per-split pair budgets")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"latent pool directory already exists: {out}; use a new directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete pool. An interrupted generation leaves a staging
    # directory, so retrying the same output path can start safely from scratch.
    working = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    manifest = {
        "format": "path-qa-latent", "version": VERSION, "seed": seed,
        "semantic_config": semantic_config(graph, task),
        "requested_query_pairs": {"train": train_pairs, "valid": valid_pairs, "test": test_pairs},
        "stats": {}, "train_checkpoints": {}, "sha256": {},
    }
    seen: set[str] = set()
    # Evaluation is reserved first; extending train cannot change either eval split.
    for split, target in (("valid", valid_pairs), ("test", test_pairs), ("train", train_pairs)):
        stats = SemanticStats(graph, task, split)
        pair_count = 0
        next_progress = 100_000
        pending = sorted({int(c) for c in checkpoints if 0 < c <= target}) if split == "train" else []
        digest = hashlib.sha256()
        with (working / f"{split}.jsonl").open("wb") as handle:
            for index, (key, sample) in enumerate(latent_stream(graph, task, seed, split, seen)):
                row = {"sample_id": f"{split}:{index}:{key}", **sample.to_dict()}
                encoded = (json.dumps(row, separators=(",", ":")) + "\n").encode()
                handle.write(encoded)
                digest.update(encoded)
                stats.update(sample)
                pair_count += len(sample.query_vertices) - 1
                if pair_count >= next_progress:
                    print(f"latent {split}: {pair_count}/{target} pairs ({stats.n} samples)", flush=True)
                    next_progress = (pair_count // 100_000 + 1) * 100_000
                while pending and pair_count >= pending[0]:
                    mark = pending.pop(0)
                    manifest["train_checkpoints"][str(mark)] = {
                        "end_sample_index_exclusive": index + 1,
                        "actual_query_pairs": pair_count, "stats": stats.report(),
                    }
                if pair_count >= target:
                    break
        manifest["stats"][split] = stats.report()
        manifest["sha256"][split] = digest.hexdigest()
        print(f"latent {split}: {stats.n} samples, {pair_count} query pairs", flush=True)
    (working / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    working.rename(out)
    return manifest


def encode_latent_pool(
    source: str | Path, language: Language, task: TaskConfig,
    train_pairs: int | None = None, encoding_seed: int = 0,
) -> tuple[Splits, dict]:
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format"), manifest.get("version")) != ("path-qa-latent", VERSION):
        raise ValueError("unsupported latent pool format/version")
    if manifest["semantic_config"] != semantic_config(language.graph, task):
        raise ValueError("latent graph/task differs from encoding config (CoT may differ)")
    available = manifest["stats"]["train"]["query_pair_occurrences"]
    if train_pairs is not None and not 0 < train_pairs <= available:
        raise ValueError(f"train pairs must lie in 1..{available}")
    encoded_splits = {}
    selection_sha256 = {}
    for split in ("train", "valid", "test"):
        samples = []
        pairs = 0
        digest = hashlib.sha256()
        selected_digest = hashlib.sha256()
        with (source / f"{split}.jsonl").open("rb") as handle:
            # Hash full source even when selecting a prefix, to detect corruption.
            for line in handle:
                digest.update(line)
                if split == "train" and train_pairs is not None and pairs >= train_pairs:
                    continue
                row = json.loads(line)
                sample = LatentQASample.from_dict(row)
                if sample_key(sample) != row["sample_id"].rsplit(":", 1)[-1]:
                    raise ValueError("latent sample content does not match its ID")
                rng = random.Random(stable_seed(VERSION, encoding_seed, row["sample_id"]))
                samples.append(encode_qa(sample, language, task, rng, row["sample_id"]))
                selected_digest.update((row["sample_id"] + "\n").encode())
                pairs += len(sample.query_vertices) - 1
        if digest.hexdigest() != manifest["sha256"][split]:
            raise ValueError(f"latent {split} checksum mismatch")
        encoded_splits[split] = tuple(samples)
        selection_sha256[split] = selected_digest.hexdigest()
    return Splits(**encoded_splits), {
        "source": str(source.resolve()), "source_sha256": manifest["sha256"],
        "encoding_seed": encoding_seed, "train_query_pairs_requested": train_pairs,
        "latent_seed": manifest["seed"], "stream_version": VERSION,
        "selection_sha256": selection_sha256,
    }
