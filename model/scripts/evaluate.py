#!/usr/bin/env python3
"""CLI: evaluate a checkpoint on a dataset (validity, loss vs floor, diversity).

    python scripts/evaluate.py outputs/smoke_micro/best.pt
    python scripts/evaluate.py outputs/smoke_micro/best.pt --dataset ../data/outputs/smoke \
        --n-samples 200 --cuts 0 50 --temperature 1.0

Writes ``eval_report.json`` next to the checkpoint (or into ``--out-dir``) and prints
a human-readable summary.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# The modules live in the parent ``model/`` directory. This script's own directory is
# removed from ``sys.path`` first: it contains a ``train.py`` that would otherwise
# shadow the library ``train`` module (the same trap as ``data/scripts/inspect.py``).
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import load_frozen, resolve_device, seed_everything  # noqa: E402
from evals import format_report, run_all  # noqa: E402
from train import load_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--dataset", default=None, help="dataset dir (default: the one in the config)")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument(
        "--gen-batch-size",
        type=int,
        default=None,
        help="number of sequences generated in parallel (default: checkpoint config)",
    )
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--cuts", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, cfg, ckpt = load_checkpoint(args.checkpoint, device)
    model.eval()

    eval_overrides = {
        k: v
        for k, v in {
            "n_samples": args.n_samples,
            "gen_batch_size": args.gen_batch_size,
            "temperature": args.temperature,
            "cuts": None if args.cuts is None else tuple(args.cuts),
            "seed": args.seed,
        }.items()
        if v is not None
    }
    eval_cfg = dataclasses.replace(cfg.eval, **eval_overrides)

    dataset_dir = Path(args.dataset) if args.dataset else cfg.resolved_dataset_dir()
    seed_everything(eval_cfg.seed)
    data = load_frozen(dataset_dir, cfg.model.context_len)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parent
    report = run_all(
        model,
        data,
        eval_cfg,
        device=device,
        out_dir=out_dir,
        extra={"checkpoint": str(Path(args.checkpoint).resolve()), "step": ckpt.get("step")},
    )
    print(f"device       {device}")
    print(f"checkpoint   {args.checkpoint}  (step {ckpt.get('step')})")
    print(format_report(report))
    print(f"report       {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
