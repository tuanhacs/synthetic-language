#!/usr/bin/env python3
"""CLI: train a model on a synthetic-language dataset.

    python scripts/train.py configs/smoke.yaml
    python scripts/train.py configs/smoke.yaml --max-steps 500 --out-dir outputs/quick
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

from config import ConfigError, ModelConfig, SIZE_PRESETS, load_config  # noqa: E402
from train import train  # noqa: E402


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="run config YAML (see configs/smoke.yaml)")
    ap.add_argument("--max-steps", type=positive_int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=positive_int, default=None)
    ap.add_argument("--eval-every", type=positive_int, default=None)
    ap.add_argument("--eval-batches", type=positive_int, default=None)
    ap.add_argument("--early-stopping-patience", type=int, default=None,
                    help="stop after this many validation checks without significant improvement; 0 disables")
    ap.add_argument("--early-stopping-min-delta", type=float, default=None, help="absolute bits/token improvement")
    ap.add_argument(
        "--resume",
        choices=("auto", "never", "require"),
        default=None,
        help="resume from out_dir/latest.pt (default comes from config)",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto-detect)")
    ap.add_argument("--size", default=None, help="override model.size preset")
    ap.add_argument("--d-model", type=positive_int, default=None, help="override model width")
    ap.add_argument("--n-layers", type=positive_int, default=None, help="override layer count")
    ap.add_argument("--n-heads", type=positive_int, default=None, help="override attention head count")
    ap.add_argument("--wandb", action="store_true", help="enable online W&B tracking")
    ap.add_argument("--wandb-mode", choices=("disabled", "online", "offline"), default=None)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-tags", nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tracking_overrides = {
        key: value for key, value in {
            "mode": args.wandb_mode if args.wandb_mode is not None else ("online" if args.wandb else None),
            "project": args.wandb_project, "entity": args.wandb_entity,
            "group": args.wandb_group, "name": args.wandb_name,
            "tags": tuple(args.wandb_tags) if args.wandb_tags is not None else None,
        }.items() if value is not None
    }
    if tracking_overrides:
        cfg = dataclasses.replace(cfg, wandb=dataclasses.replace(cfg.wandb, **tracking_overrides))
    overrides = {
        k: v
        for k, v in {
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "checkpoint_every": args.checkpoint_every,
            "eval_every": args.eval_every,
            "eval_batches": args.eval_batches,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "resume": args.resume,
            "out_dir": args.out_dir,
            "device": args.device,
        }.items()
        if v is not None
    }
    if overrides:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **overrides))
    model_cfg = cfg.model
    if args.size is not None:
        if args.size not in SIZE_PRESETS:
            ap.error(f"--size must be one of {sorted(SIZE_PRESETS)}")
        model_cfg = ModelConfig(
            size=args.size,
            context_len=model_cfg.context_len,
            vocab_size=model_cfg.vocab_size,
            dropout=model_cfg.dropout,
            **SIZE_PRESETS[args.size],
        )

    model_overrides = {
        key: value
        for key, value in {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
        }.items()
        if value is not None
    }
    if model_overrides:
        try:
            model_cfg = dataclasses.replace(model_cfg, **model_overrides)
        except ConfigError as exc:
            ap.error(str(exc))
    if model_cfg is not cfg.model:
        cfg = dataclasses.replace(cfg, model=model_cfg)

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
