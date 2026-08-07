#!/usr/bin/env python3
"""Generate a frozen dataset from a YAML config.

    python scripts/gen_dataset.py configs/smoke_4x4_prefix.yaml --out outputs/smoke

Pipeline: load config -> make base code -> certify -> power to C^x -> assign
codebooks -> certify language -> build pool -> apply noise -> split -> save.
Certification failures abort the run: a dataset must never claim a regime it
does not have.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Put the package root on sys.path and drop this scripts/ directory from it,
# so that scripts/inspect.py cannot shadow the stdlib "inspect" module.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthdata.codebooks import assign  # noqa: E402
from synthdata.codes import certify as certify_code, make_code  # noqa: E402
from synthdata.config import Config, config_hash, load_config, validate_config  # noqa: E402
from synthdata.dataset import build_pool, pool_stats, split_pool  # noqa: E402
from synthdata.graphs import graph_from_spec  # noqa: E402
from synthdata.language import Language  # noqa: E402
from synthdata.noise import apply_noise  # noqa: E402
from synthdata.storage import save_dataset  # noqa: E402


def derive_rngs(seed: int) -> dict[str, random.Random]:
    """Independent, reproducible sub-streams for each pipeline stage."""
    master = random.Random(seed)
    return {
        name: random.Random(master.randrange(2**63))
        for name in ("code", "assignment", "pool", "noise", "split")
    }


def build(cfg: Config, pool_tokens: int | None, entropy_samples: int, quiet: bool = False):
    def log(*args: object) -> None:
        if not quiet:
            print(*args)

    for warning in validate_config(cfg):
        log(f"[warn] {warning}")

    rngs = derive_rngs(cfg.data.seed)
    graph = graph_from_spec(cfg.language.graph)

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

    codebooks = assign(cfg.language, graph, pool_code, rngs["assignment"])
    language = Language(graph=graph, codebooks=codebooks, walk_len=cfg.data.walk_len)

    report = language.certify(cfg.language.code.type)
    log("\nlanguage certification:")
    log(f"  code:      {report.code.to_dict()}")
    log(f"  codebooks: {report.codebooks.to_dict()}")
    log(f"  regime: {report.regime} | unique_decoding: {report.unique_decoding}")
    for note in report.notes:
        log(f"  [note] {note}")
    if not report.unique_decoding:
        raise SystemExit("ABORT: language does not guarantee unique decoding")

    pool = build_pool(language, cfg.data, rngs["pool"], pool_tokens=pool_tokens)
    log(f"\npool: {len(pool)} unique sentences, {sum(len(s.bits) for s in pool)} bits")

    if cfg.data.noise is not None:
        pool = apply_noise(pool, cfg.data.noise, rngs["noise"], language=language)
        n_noised = sum(1 for s in pool if s.noised_bits is not None)
        log(f"noise {cfg.data.noise.to_dict()}: {n_noised}/{len(pool)} sentences corrupted")

    splits = split_pool(pool, cfg.data.split, rng=rngs["split"])
    log(f"splits: {splits.sizes()} | bits {splits.total_bits()}")

    extra: dict[str, object] = {}
    if entropy_samples > 0:
        held_out = (splits.valid or splits.train)[:entropy_samples]
        floor = language.entropy_floor(held_out)
        extra["entropy_floor_bits_per_token"] = floor
        extra["entropy_floor_samples"] = len(held_out)
        log(f"entropy floor: {floor:.4f} bits/token (on {len(held_out)} held-out sentences)")
    return language, splits, report, base, extra


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to the YAML config")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument(
        "--pool-tokens",
        type=int,
        default=None,
        help="override data.pool_tokens (for quick smoke runs; the config keeps its value)",
    )
    ap.add_argument(
        "--entropy-samples",
        type=int,
        default=64,
        help="held-out sentences used for the entropy floor (0 = skip)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    print(f"config: {args.config}  hash={config_hash(cfg)}")
    print(f"  language: {cfg.language.to_dict()}")
    print(f"  data:     {cfg.data.to_dict()}")
    if args.pool_tokens is not None:
        print(f"  [override] pool_tokens = {args.pool_tokens} (config value kept in the manifest)")

    language, splits, report, base, extra = build(cfg, args.pool_tokens, args.entropy_samples)
    if args.pool_tokens is not None:
        extra["pool_tokens_override"] = args.pool_tokens

    out = save_dataset(
        args.out, cfg, language, splits, report=report, base_code=base.to_list(), extra=extra
    )
    print(f"\nwritten to {out.resolve()}")
    for name, split in splits.items():
        stats = pool_stats(split)
        if stats["num_sentences"]:
            print(
                f"  {name:5s} {stats['num_sentences']:7d} sentences  "
                f"{stats['total_bits']:9d} bits  "
                f"len {stats['sentence_bits']['min']}..{stats['sentence_bits']['max']} "
                f"(mean {stats['sentence_bits']['mean']:.1f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
