#!/usr/bin/env python3
"""CLI: sample sentences from a checkpoint and decode them with the exact decoder.

    python scripts/generate.py outputs/smoke_micro/best.pt -n 5
    python scripts/generate.py outputs/smoke_micro/best.pt -n 5 --cut 50 --temperature 0.8

Each sample is printed with its decoded walk (``synthdata``'s ``Language.decode``) or
``INVALID``. ``--score`` adds aggregate validity / diversity numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# The modules live in the parent ``model/`` directory. This script's own directory is
# removed from ``sys.path`` first: it contains a ``train.py`` that would otherwise
# shadow the library ``train`` module (the same trap as ``data/scripts/inspect.py``).
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import load_frozen, resolve_device, seed_everything  # noqa: E402
from evals import eval_diversity  # noqa: E402
from sample import generate  # noqa: E402
from train import load_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("-n", "--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cut", type=int, default=0, help="complete the first CUT bits of test sentences")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", action="store_true", help="also print validity / diversity summary")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-bits", type=int, default=120, help="truncate printed bit strings")
    args = ap.parse_args()

    device = resolve_device(args.device)
    seed_everything(args.seed)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    model.eval()

    dataset_dir = Path(args.dataset) if args.dataset else cfg.resolved_dataset_dir()
    data = load_frozen(dataset_dir, cfg.model.context_len)
    language = data.language

    prefixes = None
    if args.cut:
        pool = [s.bits for s in data.test_samples if len(s.bits) > args.cut]
        if not pool:
            ap.error(f"no test sentence longer than --cut {args.cut}")
        prefixes = [pool[i % len(pool)][: args.cut] for i in range(args.n_samples)]

    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    bits = generate(
        model,
        data.tokenizer,
        n=args.n_samples,
        temperature=args.temperature,
        max_len=args.max_len,
        prefix_bits=prefixes,
        device=device,
        rng=rng,
    )

    print(f"device {device}   tau={args.temperature}   cut={args.cut}   n={args.n_samples}")
    train_bits = {s.bits for s in data.train_samples}
    for i, string in enumerate(bits):
        shown = string if len(string) <= args.max_bits else string[: args.max_bits] + "..."
        walks = language.decode(string)
        if walks:
            walk = " ".join(str(v) for v in walks[0])
            tag = f"walk[{len(walks[0])}] {walk}"
            if len(walks) > 1:
                tag += f"   (+{len(walks) - 1} more walks)"
            if string in train_bits:
                tag += "   [IN TRAIN]"
        else:
            tag = "INVALID"
        print(f"[{i}] {len(string):>4} bits  {shown}")
        print(f"     {tag}")

    if args.score:
        valid = sum(1 for s in bits if language.is_valid(s))
        div = eval_diversity(bits, train_bits)
        print(
            f"\nvalid {100.0 * valid / len(bits):.1f}%   "
            f"distinct {div['distinct_frac']:.3f}   memorised {div['memorisation_frac']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
