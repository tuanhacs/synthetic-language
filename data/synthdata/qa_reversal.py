"""Exact prefix/suffix reversal of encoded QA, including segment rationales."""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from pathlib import Path

from .codebooks import Codebooks
from .config import Config
from .dataset import Splits
from .language import Language
from .qa import LatentQASample, QASample, encode_qa, qa_pool_stats
from .qa_latent import VERSION, sample_key, semantic_stats, stable_seed
from .storage import load_dataset, save_dataset


def reverse_cuts(cuts: tuple[int, ...]) -> tuple[int, ...]:
    total = cuts[-1]
    return tuple(total - cut for cut in reversed(cuts[:-1])) + (total,)


def reverse_sample(sample: QASample) -> QASample:
    """Reverse vertices and individual bit fields, not the QA field order."""
    n = len(sample.answer_walk)
    return QASample(
        query_bits=sample.query_bits[::-1], answer_bits=sample.answer_bits[::-1],
        query_vertices=sample.query_vertices[::-1], answer_walk=sample.answer_walk[::-1],
        query_cuts=reverse_cuts(sample.query_cuts), answer_cuts=reverse_cuts(sample.answer_cuts),
        segment_cuts=tuple(n + 1 - cut for cut in reversed(sample.segment_cuts[:-1])) + (n,),
        cot_segments_bits=tuple(bits[::-1] for bits in reversed(sample.cot_segments_bits)),
        sample_id=sample.sample_id,  # source ID intentionally retained for pairing
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_reversed_dataset(cfg: Config, source: Path, latent_pool: Path, out: Path,
                           encoding_seed: int = 0) -> None:
    if out.exists():
        raise ValueError(f"refusing to overwrite {out}")
    original = load_dataset(source)
    if (cfg.task != original.config.task or cfg.task.type != "path-qa" or
            cfg.language.code.type != "suffix-free" or
            original.config.language.code.type != "prefix-free"):
        raise ValueError("requires matching path-QA tasks and prefix -> suffix configs")
    target_language = cfg.language.to_dict()
    source_language = original.config.language.to_dict()
    target_language.pop("code")
    source_language.pop("code")
    if target_language != source_language or cfg.data != original.config.data:
        raise ValueError("graph/assignment/data settings must match the prefix source")
    if cfg.task.pair_direction != "unordered" and cfg.task.split_mode == "held-out-pairs":
        raise ValueError("reversal of held-out pairs requires unordered pairs")
    graph = original.language.graph
    books = Codebooks(tuple(tuple(word[::-1] for word in book)
                            for book in original.language.codebooks.books))
    language = Language(graph, books, original.language.walk_len)
    report = language.certify("suffix-free")
    if not report.code.suffix_free or not report.unique_decoding:
        raise ValueError("reversed language failed suffix/unique-decoding certification")
    provenance = original.manifest["stats"]["latent_pool"]
    pool_manifest = json.loads((latent_pool / "manifest.json").read_text())
    if (provenance["source_sha256"] != pool_manifest["sha256"] or
            provenance["encoding_seed"] != encoding_seed):
        raise ValueError("prefix source differs from the requested latent pool/encoding seed")

    # Remove natural-test collisions against reversed *full* train/valid pool,
    # not merely the selected train subset: test stays fixed across sweep sizes.
    test_keys = {sample_key(s) for s in original.splits.test}
    collisions = set()
    for split in ("train", "valid"):
        digest = hashlib.sha256()
        with (latent_pool / f"{split}.jsonl").open("rb") as handle:
            for line in handle:
                digest.update(line)
                row = LatentQASample.from_dict(json.loads(line))
                key = sample_key(LatentQASample(row.query_vertices[::-1], row.answer_walk[::-1], ()))
                if key in test_keys:
                    collisions.add(key)
        if digest.hexdigest() != pool_manifest["sha256"][split]:
            raise ValueError(f"latent {split} checksum mismatch")
    train = tuple(map(reverse_sample, original.splits.train))
    valid = tuple(map(reverse_sample, original.splits.valid))
    paired = tuple(map(reverse_sample, original.splits.test))
    natural = tuple(
        encode_qa(LatentQASample(s.query_vertices, s.answer_walk, s.segment_cuts), language, cfg.task,
                  random.Random(stable_seed(VERSION, encoding_seed, s.sample_id)), s.sample_id)
        for s in original.splits.test if sample_key(s) not in collisions
    )
    if not natural:
        raise ValueError("natural test is empty after removing reversed-training collisions")
    splits = Splits(train, valid, natural)
    observed = max(len(s.bits) + 2 for split in (train, valid, natural, paired) for s in split)
    extra = {
        "latent_pool": provenance,
        "reversal": {
            "version": 1, "prefix_source": str(source.resolve()),
            "prefix_sha256": {name: _sha(source / name) for name in
                              ("codebook.json", "train.jsonl", "valid.jsonl", "test.jsonl")},
            "natural_test_collisions_removed": len(collisions),
            "test_mode": "natural", "sample_id_semantics": "prefix source identity",
        },
        "observed_max_sequence_tokens": observed,
        "recommended_context_len": ((observed + 63) // 64) * 64,
        "semantic_splits": {name: semantic_stats(samples, graph, cfg.task, name)
                            for name, samples in splits.items()},
    }
    # Evaluation loads test only, so this variant needs no duplicate train/valid.
    paired_extra = {**extra, "reversal": {**extra["reversal"], "test_mode": "reversed-prefix"},
                    "semantic_splits": {"test": semantic_stats(paired, graph, cfg.task, "test")}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qa-reversal-", dir=out.parent) as temporary:
        staging = Path(temporary) / "dataset"
        save_dataset(staging, cfg, language, splits, report=report, extra=extra)
        save_dataset(staging / "test_reversed", cfg, language, Splits((), (), paired),
                     report=report, extra=paired_extra)
        # Fingerprints distinguish natural/paired views and source bytes.
        for directory in (staging, staging / "test_reversed"):
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["dataset_fingerprint"] = hashlib.sha256(json.dumps({
                "base": manifest["dataset_fingerprint"], "reversal": manifest["stats"]["reversal"],
            }, sort_keys=True).encode()).hexdigest()
            path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(out)
    print(f"Reversed suffix written to {out}")
    for name, samples in (*splits.items(), ("test_reversed", paired)):
        print(f"  {name}: {len(samples)} examples, {qa_pool_stats(samples)['query_pair_occurrences']} query pairs")
    print(f"Natural test collisions removed: {len(collisions)}; context_len={extra['recommended_context_len']}")
