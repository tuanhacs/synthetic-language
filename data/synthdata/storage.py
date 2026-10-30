"""On-disk dataset format.

A dataset directory contains::

    manifest.json     config + config_hash + certification report + stats
    codebook.json     graph + every B_v (enough to rebuild the Language alone)
    train.jsonl       one LM Sample or path-QA QASample record per line
    valid.jsonl
    test.jsonl

``codebook.json`` is intentionally self-sufficient: reconstructing the language
never requires re-running the (version-dependent) generators.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .codebooks import Codebooks
from .config import Config, config_hash, parse_config
from .dataset import Splits, pool_stats
from .graphs import GridGraph, graph_from_dict
from .language import Language, LanguageReport, Sample
from .qa import QASample, qa_pool_stats

MANIFEST = "manifest.json"
CODEBOOK = "codebook.json"
SPLIT_FILES = {"train": "train.jsonl", "valid": "valid.jsonl", "test": "test.jsonl"}


@dataclass(frozen=True)
class LoadedDataset:
    config: Config
    language: Language
    splits: Splits
    manifest: dict[str, Any]


def _write_jsonl(path: Path, samples: Iterable[Sample | QASample]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.to_dict(), separators=(",", ":")))
            fh.write("\n")


def _read_jsonl(path: Path, qa: bool = False) -> tuple[Sample | QASample, ...]:
    if not path.exists():
        return ()
    with path.open("r", encoding="utf-8") as fh:
        sample_type = QASample if qa else Sample
        return tuple(sample_type.from_dict(json.loads(line)) for line in fh if line.strip())


def save_dataset(
    out_dir: str | Path,
    cfg: Config,
    language: Language,
    splits: Splits,
    report: LanguageReport | None = None,
    base_code: Sequence[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write manifest, codebook and the three split files. Returns the directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = report or language.certify(cfg.language.code.type)

    codebook_doc = {
        "graph": language.graph.to_dict(),
        "codebooks": language.codebooks.to_dict(),
        "walk_len": list(language.walk_len),
        "base_code": list(base_code) if base_code is not None else None,
    }
    (out / CODEBOOK).write_text(json.dumps(codebook_doc, indent=2) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "format_version": 1,
        "config": cfg.to_dict(),
        "config_hash": config_hash(cfg),
        "certification": report.to_dict(),
        "stats": {
            "splits": {
                name: (qa_pool_stats(split) if cfg.task.type == "path-qa" else pool_stats(split))
                for name, split in splits.items()
            },
            "codeword_pool_size": len(language.codebooks.global_code()),
            "codeword_length_histogram": {
                str(length): count
                for length, count in language.codebooks.global_code().length_histogram().items()
            },
        },
        "files": dict(SPLIT_FILES, codebook=CODEBOOK),
    }
    if extra:
        manifest["stats"].update(extra)
        if extra.get("latent_pool") is not None:
            provenance = extra["latent_pool"]
            identity = {
                "config_hash": manifest["config_hash"],
                "codebook": codebook_doc,
                "selected_samples": provenance["selection_sha256"],
                "encoding_seed": provenance["encoding_seed"],
            }
            manifest["dataset_fingerprint"] = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for name, split in splits.items():
        _write_jsonl(out / SPLIT_FILES[name], split)
    return out


def load_codebook(dir_path: str | Path) -> tuple[GridGraph, Codebooks, tuple[int, int]]:
    """Load graph + codebooks + walk length range from ``codebook.json``."""
    doc = json.loads((Path(dir_path) / CODEBOOK).read_text(encoding="utf-8"))
    graph = graph_from_dict(doc["graph"])
    books = Codebooks.from_dict(doc["codebooks"])
    walk_len = tuple(int(x) for x in doc["walk_len"])
    return graph, books, (walk_len[0], walk_len[1])


def load_dataset(
    dir_path: str | Path,
    load_splits: bool = True,
    split_names: Sequence[str] | None = None,
) -> LoadedDataset:
    """Rebuild a dataset, optionally reading only selected JSONL splits.

    ``split_names`` keeps evaluation-only callers from parsing large train and
    validation files they will never use. Existing callers still load all
    splits by default.
    """
    path = Path(dir_path)
    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    cfg = parse_config(manifest["config"])
    graph, books, walk_len = load_codebook(path)
    language = Language(graph=graph, codebooks=books, walk_len=walk_len)
    if load_splits:
        qa = cfg.task.type == "path-qa"
        requested = set(SPLIT_FILES) if split_names is None else set(split_names)
        unknown = requested - set(SPLIT_FILES)
        if unknown:
            raise ValueError(f"unknown dataset splits: {sorted(unknown)}")
        splits = Splits(
            train=(
                _read_jsonl(path / SPLIT_FILES["train"], qa=qa)
                if "train" in requested else ()
            ),
            valid=(
                _read_jsonl(path / SPLIT_FILES["valid"], qa=qa)
                if "valid" in requested else ()
            ),
            test=(
                _read_jsonl(path / SPLIT_FILES["test"], qa=qa)
                if "test" in requested else ()
            ),
        )
    else:
        splits = Splits(train=(), valid=(), test=())
    return LoadedDataset(config=cfg, language=language, splits=splits, manifest=manifest)
