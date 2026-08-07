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

from config import load_config  # noqa: E402
from train import train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="run config YAML (see configs/smoke.yaml)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto-detect)")
    ap.add_argument("--size", default=None, help="override model.size preset")
    args = ap.parse_args()

    cfg = load_config(args.config)
    overrides = {
        k: v
        for k, v in {
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "out_dir": args.out_dir,
            "device": args.device,
        }.items()
        if v is not None
    }
    if overrides:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **overrides))
    if args.size is not None:
        from config import SIZE_PRESETS, ModelConfig

        if args.size not in SIZE_PRESETS:
            ap.error(f"--size must be one of {sorted(SIZE_PRESETS)}")
        cfg = dataclasses.replace(
            cfg,
            model=ModelConfig(
                size=args.size,
                context_len=cfg.model.context_len,
                vocab_size=cfg.model.vocab_size,
                dropout=cfg.model.dropout,
                **SIZE_PRESETS[args.size],
            ),
        )

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
