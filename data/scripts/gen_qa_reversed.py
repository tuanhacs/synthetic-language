#!/usr/bin/env python3
"""Build suffix train/valid by exact prefix reversal and two suffix test views."""
import argparse
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthdata.config import load_config
from synthdata.qa_reversal import build_reversed_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--prefix-dataset", type=Path, required=True)
    ap.add_argument("--latent-pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encoding-seed", type=int, default=0)
    args = ap.parse_args()
    build_reversed_dataset(load_config(args.config), args.prefix_dataset,
                           args.latent_pool, args.out, args.encoding_seed)


if __name__ == "__main__":
    main()
