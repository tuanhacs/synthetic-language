#!/usr/bin/env python3
"""Generate a frozen dataset from a YAML config.

    python scripts/gen_dataset.py configs/smoke_4x4_prefix.yaml --out outputs/smoke

Pipeline: load config -> generate C^x or import a codeword pool -> assign
codebooks to vertices -> certify language -> build data -> split -> save.
Certification failures abort the run: a dataset must never claim a regime it
does not have.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Put the package root on sys.path and drop this scripts/ directory from it,
# so that scripts/inspect.py cannot shadow the stdlib "inspect" module.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthdata.codebooks import Codebooks, assign, remap_assignment  # noqa: E402
from synthdata.codes import Code, certify as certify_code, load_codeword_pool, make_code  # noqa: E402
from synthdata.config import Config, config_hash, load_config, validate_config  # noqa: E402
from synthdata.dataset import build_pool, pool_stats, split_pool  # noqa: E402
from synthdata.graphs import graph_from_spec  # noqa: E402
from synthdata.language import Language  # noqa: E402
from synthdata.noise import apply_noise  # noqa: E402
from synthdata.qa import build_qa_splits, qa_pool_stats  # noqa: E402
from synthdata.qa_latent import encode_latent_pool, semantic_stats, frequency_report_lines  # noqa: E402
from synthdata.storage import save_dataset  # noqa: E402


def derive_rngs(seed: int) -> dict[str, random.Random]:
    """Independent, reproducible sub-streams for each pipeline stage."""
    master = random.Random(seed)
    return {
        name: random.Random(master.randrange(2**63))
        for name in ("code", "assignment", "pool", "noise", "split")
    }


def _resolve_input_file(raw_path: str, config_path: str | Path | None) -> Path:
    """Resolve an external input relative to its YAML config file."""
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw
    base = Path(config_path).resolve().parent if config_path is not None else Path.cwd()
    return (base / raw).resolve()


def build(
    cfg: Config,
    pool_tokens: int | None,
    pool_query_pairs: int | None,
    entropy_samples: int,
    quiet: bool = False,
    config_path: str | Path | None = None,
    latent_pool: str | Path | None = None,
    train_query_pairs: int | None = None,
    encoding_seed: int = 0,
    assignment_template: str | Path | None = None,
):
    def log(*args: object) -> None:
        if not quiet:
            print(*args)

    for warning in validate_config(cfg):
        log(f"[warn] {warning}")

    rngs = derive_rngs(cfg.data.seed)
    graph = graph_from_spec(cfg.language.graph)
    base: Code | None
    source_pool_path: Path | None = None

    if cfg.language.codeword_pool_file is not None:
        source_pool_path = _resolve_input_file(cfg.language.codeword_pool_file, config_path)
        try:
            pool_code = load_codeword_pool(source_pool_path)
        except ValueError as exc:
            raise SystemExit(f"ABORT: {exc}") from None
        base = None
        log(f"\nimported codeword pool: {source_pool_path}")
        log(
            f"  words={len(pool_code)} lengths={pool_code.min_len}..{pool_code.max_len} bits"
        )
    else:
        base = make_code(cfg.language.code, rngs["code"])
        base_report = certify_code(base)
        log(f"\nbase code ({cfg.language.code.type}): {base.to_list()}")
        log(f"  certified: {base_report.to_dict()}")
        if not base_report.matches(cfg.language.code.type):
            raise SystemExit(
                f"ABORT: base code fails certification for type {cfg.language.code.type!r} "
                f"(certified regime: {base_report.regime})"
            )

        pool_code = base.power(cfg.language.code.power_x)
        log(
            f"\ncodeword pool C^{cfg.language.code.power_x}: |C^x| = {len(pool_code)}, "
            f"lengths {pool_code.min_len}..{pool_code.max_len} bits"
        )

    k_lo, k_hi = cfg.language.k_range
    max_needed = graph.num_vertices * k_hi
    if len(pool_code) < max_needed:
        raise SystemExit(
            f"ABORT: codeword pool has {len(pool_code)} words, need up to {max_needed} "
            f"for {graph.num_vertices} vertices x k_max={k_hi}"
        )
    if source_pool_path is not None and len(pool_code) < 4 * max_needed:
        log(
            f"[warn] pool slack is small: {len(pool_code)} words vs maximum required {max_needed}"
        )
    try:
        if assignment_template is None:
            codebooks = assign(cfg.language, graph, pool_code, rngs["assignment"])
        else:
            template_doc = json.loads(Path(assignment_template).read_text(encoding="utf-8"))
            if template_doc.get("graph") != graph.to_dict():
                raise ValueError("assignment template graph differs from config")
            template = Codebooks.from_dict(template_doc["codebooks"])
            codebooks = remap_assignment(template, cfg.language, graph, pool_code, rngs["assignment"])
    except ValueError as exc:
        raise SystemExit(f"ABORT: codebook assignment failed: {exc}") from None
    counts = codebooks.k_per_vertex
    log(
        f"assigned codewords: total={sum(counts)} per-vertex={min(counts)}..{max(counts)}"
    )

    language = Language(graph=graph, codebooks=codebooks, walk_len=cfg.data.walk_len)

    max_word_len = language.codebooks.global_code().max_len
    if cfg.task.type == "path-qa":
        max_waypoints = cfg.task.query_len[1]
        # Adjacent answer segments share their boundary waypoint.
        max_answer_vertices = 1 + (max_waypoints - 1) * (cfg.task.segment_len[1] - 1)
        if cfg.task.cot_mode == "segments":
            max_cot_vertices = (max_waypoints - 1) * cfg.task.segment_len[1]
            separators = max_waypoints
        else:
            max_cot_vertices = 0
            separators = 1
        worst_bits = (
            (max_waypoints + max_cot_vertices + max_answer_vertices) * max_word_len
            + separators
            + 2  # BOS/EOS
        )
    else:
        worst_bits = cfg.data.walk_len[1] * max_word_len + 2
    if cfg.data.context_len is not None and worst_bits > cfg.data.context_len:
        log(
            f"[warn] actual worst-case sentence length {worst_bits} bits exceeds "
            f"context_len {cfg.data.context_len}"
        )

    report = language.certify(cfg.language.code.type)
    log("\nlanguage certification:")
    log(f"  code:      {report.code.to_dict()}")
    log(f"  codebooks: {report.codebooks.to_dict()}")
    log(f"  regime: {report.regime} | unique_decoding: {report.unique_decoding}")
    for note in report.notes:
        log(f"  [note] {note}")
    if source_pool_path is not None and not report.code.matches(cfg.language.code.type):
        raise SystemExit(
            f"ABORT: assigned words do not match declared type "
            f"{cfg.language.code.type!r} (certified regime: {report.code.regime})"
        )
    if not report.unique_decoding:
        raise SystemExit("ABORT: language does not guarantee unique decoding")

    latent_provenance = None
    if latent_pool is not None:
        if cfg.task.type != "path-qa":
            raise ValueError("--latent-pool requires task.type: path-qa")
        splits, latent_provenance = encode_latent_pool(
            latent_pool, language, cfg.task, train_query_pairs, encoding_seed
        )
        pool = [sample for _, split in splits.items() for sample in split]
    elif cfg.task.type == "path-qa":
        splits = build_qa_splits(
            language,
            cfg.task,
            cfg.data,
            rngs["pool"],
            pool_tokens=pool_tokens,
            pool_query_pairs=pool_query_pairs,
        )
        pool = [sample for _, split in splits.items() for sample in split]
    else:
        if pool_query_pairs is not None:
            raise ValueError("--pool-query-pairs requires task.type: path-qa")
        pool = build_pool(language, cfg.data, rngs["pool"], pool_tokens=pool_tokens)
        if cfg.data.noise is not None:
            pool = apply_noise(pool, cfg.data.noise, rngs["noise"], language=language)
            n_noised = sum(1 for s in pool if s.noised_bits is not None)
            log(f"noise {cfg.data.noise.to_dict()}: {n_noised}/{len(pool)} sentences corrupted")
        splits = split_pool(pool, cfg.data.split, rng=rngs["split"])
    log(f"\npool: {len(pool)} unique examples, {sum(len(s.bits) for s in pool)} tokens")
    log(f"splits: {splits.sizes()} | bits {splits.total_bits()}")
    qa_stats_by_split = None
    if cfg.task.type == "path-qa":
        qa_stats_by_split = {
            name: qa_pool_stats(split) for name, split in splits.items()
        }
        log(
            "query pairs: "
            + str(
                {
                    name: stats["query_pair_occurrences"]
                    for name, stats in qa_stats_by_split.items()
                }
            )
        )
        observed_max = max(
            stats["full_sequence_tokens"]["max"] or 0
            for stats in qa_stats_by_split.values()
        )
        recommended_context = ((observed_max + 63) // 64) * 64
        log("full sequence tokens (BOS + sample + EOS):")
        for name, stats in qa_stats_by_split.items():
            log(f"  {name:5s} {stats['full_sequence_tokens']}")
        log(
            f"observed max={observed_max}; recommended context_len="
            f"{recommended_context} (rounded up to a multiple of 64)"
        )

    extra: dict[str, object] = {}
    if cfg.task.type == "path-qa":
        extra["semantic_splits"] = {
            name: semantic_stats(split, graph, cfg.task, name)
            for name, split in splits.items()
        }
        for name, stats in extra["semantic_splits"].items():
            pairs = stats["query_pairs_directed"]
            edges = stats["graph_edges_undirected"]
            log(f"{name}: query-length histogram={stats['histograms']['query_vertices']}")
            log(f"  query-pair coverage {pairs['seen']}/{pairs['eligible']}; "
                f"graph-edge coverage {edges['seen']}/{edges['eligible']}")
            for line in frequency_report_lines(stats):
                log(line)
    if latent_provenance is not None:
        extra["latent_pool"] = latent_provenance
    if assignment_template is not None:
        extra["assignment_template"] = str(Path(assignment_template).resolve())
    if qa_stats_by_split is not None:
        extra["observed_max_sequence_tokens"] = observed_max
        extra["recommended_context_len"] = recommended_context
    if source_pool_path is not None:
        extra["codeword_pool_file"] = cfg.language.codeword_pool_file
        extra["source_codeword_pool_size"] = len(pool_code)
    if entropy_samples > 0 and cfg.task.type != "path-qa":
        held_out = (splits.valid or splits.train)[:entropy_samples]
        floor = language.entropy_floor(held_out)
        extra["entropy_floor_bits_per_token"] = floor
        extra["entropy_floor_samples"] = len(held_out)
        log(f"entropy floor: {floor:.4f} bits/token (on {len(held_out)} held-out sentences)")
    elif entropy_samples > 0:
        log("[note] entropy floor is not defined for answer-conditioned path-QA")
    return language, splits, report, base, extra


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to the YAML config")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--latent-pool", help="encode an existing gen_qa_latent.py pool")
    ap.add_argument("--train-query-pairs", type=int, help="select a train prefix from --latent-pool")
    ap.add_argument("--encoding-seed", type=int, default=0)
    ap.add_argument("--assignment-template", help="codebook.json whose vertex slots/overlap supports to preserve")
    budget_group = ap.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--pool-tokens",
        type=int,
        default=None,
        help="override data.pool_tokens (for quick smoke runs; the config keeps its value)",
    )
    budget_group.add_argument(
        "--pool-query-pairs",
        type=int,
        default=None,
        help=(
            "override data.pool_query_pairs for a quick QA run "
            "(the config value is kept in the manifest)"
        ),
    )
    ap.add_argument(
        "--entropy-samples",
        type=int,
        default=64,
        help="held-out sentences used for the entropy floor (0 = skip)",
    )
    args = ap.parse_args(argv)
    if args.train_query_pairs is not None and not args.latent_pool:
        ap.error("--train-query-pairs requires --latent-pool")
    if args.latent_pool and (args.pool_tokens is not None or args.pool_query_pairs is not None):
        ap.error("latent mode uses --train-query-pairs, not --pool-tokens/--pool-query-pairs")

    cfg = load_config(args.config)
    print(f"config: {args.config}  hash={config_hash(cfg)}")
    print(f"  language: {cfg.language.to_dict()}")
    print(f"  data:     {cfg.data.to_dict()}")
    if cfg.task.type == "path-qa":
        print(f"  task:     {cfg.task.to_dict()}")
    if args.pool_tokens is not None:
        print(f"  [override] pool_tokens = {args.pool_tokens} (config value kept in the manifest)")
    if args.pool_query_pairs is not None:
        print(
            f"  [override] pool_query_pairs = {args.pool_query_pairs} "
            "(config value kept in the manifest)"
        )

    language, splits, report, base, extra = build(
        cfg,
        args.pool_tokens,
        args.pool_query_pairs,
        args.entropy_samples,
        config_path=args.config,
        latent_pool=args.latent_pool,
        train_query_pairs=args.train_query_pairs,
        encoding_seed=args.encoding_seed,
        assignment_template=args.assignment_template,
    )
    if args.pool_tokens is not None:
        extra["pool_tokens_override"] = args.pool_tokens
    if args.pool_query_pairs is not None:
        extra["pool_query_pairs_override"] = args.pool_query_pairs

    out = save_dataset(
        args.out,
        cfg,
        language,
        splits,
        report=report,
        base_code=None if base is None else base.to_list(),
        extra=extra,
    )
    print(f"\nwritten to {out.resolve()}")
    for name, split in splits.items():
        stats = qa_pool_stats(split) if cfg.task.type == "path-qa" else pool_stats(split)
        if stats["num_sentences"]:
            if cfg.task.type == "path-qa":
                print(
                    f"  {name:5s} {stats['num_sentences']:7d} examples  "
                    f"{stats['total_bits']:9d} tokens  "
                    f"{stats['query_pair_occurrences']:9d} query pairs  answer "
                    f"{stats['answer_bits']['min']}..{stats['answer_bits']['max']} bits  "
                    f"full-seq max={stats['full_sequence_tokens']['max']} tokens"
                )
            else:
                print(
                    f"  {name:5s} {stats['num_sentences']:7d} sentences  "
                    f"{stats['total_bits']:9d} bits  "
                    f"len {stats['sentence_bits']['min']}..{stats['sentence_bits']['max']} "
                    f"(mean {stats['sentence_bits']['mean']:.1f})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
