#!/usr/bin/env python3
"""Analyze region transfer and confidence for a cross-region QA checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_region import analyze_cross_region, save_cross_region_analysis  # noqa: E402
from data import load_frozen, resolve_device, seed_everything  # noqa: E402
from train import load_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--rollouts-per-query", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--confidence-queries", type=int, default=200)
    parser.add_argument("--intervention-rollouts", type=int, default=8)
    parser.add_argument("--gen-batch-size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    for name in (
        "n_queries", "rollouts_per_query", "query_batch_size",
        "gen_batch_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("confidence_queries", "intervention_rollouts"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")

    device = resolve_device(args.device)
    model, cfg, checkpoint = load_checkpoint(args.checkpoint, device)
    model.eval()
    data = load_frozen(args.dataset, cfg.model.context_len)
    seed_everything(args.seed)
    report, queries = analyze_cross_region(
        model,
        data,
        n_queries=args.n_queries,
        rollouts_per_query=args.rollouts_per_query,
        query_batch_size=args.query_batch_size,
        temperature=args.temperature,
        max_len=args.max_len,
        gen_batch_size=args.gen_batch_size,
        confidence_queries=args.confidence_queries,
        intervention_rollouts=args.intervention_rollouts,
        seed=args.seed,
        device=device,
    )
    report.update(
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_step=checkpoint.get("step"),
        dataset=str(Path(args.dataset).resolve()),
        model=cfg.model.to_dict(),
    )
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parent
    report_path, queries_path = save_cross_region_analysis(out_dir, report, queries)

    overall = report["overall"]
    print(f"device              {device}")
    print(f"queries/rollouts    {report['n_queries']} x {report['rollouts_per_query']}")
    print(f"enter overlap       {overall['entered_overlap_pct']:.2f}%")
    print(f"reach target region {overall['reached_target_region_pct']:.2f}%")
    print(f"hit destination     {overall['hit_destination_pct']:.2f}%")
    print(f"end at destination  {overall['end_at_destination_pct']:.2f}%")
    print(f"semantic success    {overall['semantic_success_pct']:.2f}%")
    print(f"destination effect  {report['destination_intervention'].get('destination_influence_score')}")
    print(f"report              {report_path}")
    print(f"per-query           {queries_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
