#!/usr/bin/env python3
"""Decode a bit string with a dataset's language (the exact scoring instrument).

    python scripts/decode.py --dataset outputs/smoke --bits 0101...
    python scripts/decode.py --dataset outputs/smoke --sample 3   # decode dataset samples

Prints every consistent walk, or INVALID if the string encodes no walk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Put the package root on sys.path and drop this scripts/ directory from it,
# so that scripts/inspect.py cannot shadow the stdlib "inspect" module.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthdata.storage import load_codebook, load_dataset  # noqa: E402
from synthdata.language import Language  # noqa: E402


def show(lang: Language, bits: str, truth: tuple[int, ...] | None = None, oracle: bool = False) -> bool:
    print(f"\nbits ({len(bits)}): {bits}")
    walks = lang.decode(bits)
    if not walks:
        print("  INVALID (encodes no walk)")
        return False
    print(f"  VALID — {len(walks)} consistent walk(s)"
          f"{' (capped)' if len(walks) >= lang.max_walks else ''}")
    for i, walk in enumerate(walks[:5]):
        print(f"    [{i}] len={len(walk)} {walk}")
        if i == 0:
            words = lang.segment(bits, walk)
            if words is not None:
                print(f"        segmentation: {' '.join(words)}")
    if len(walks) > 5:
        print(f"    ... {len(walks) - 5} more")
    if truth is not None:
        match = len(walks) == 1 and tuple(walks[0]) == truth
        print(f"  ground truth: {truth}  -> {'MATCH' if match else 'MISMATCH'}")
    if oracle:
        dist = lang.next_bit_dist(bits)
        print(f"  oracle next-token: P(0)={dist['0']:.4f} P(1)={dist['1']:.4f} P(EOS)={dist['EOS']:.4f}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset directory")
    ap.add_argument("--bits", help="bit string to decode")
    ap.add_argument("--sample", type=int, default=0, help="instead decode N test sentences")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--oracle", action="store_true", help="also print the exact next-token distribution")
    args = ap.parse_args(argv)

    if not args.bits and not args.sample:
        ap.error("give --bits or --sample N")

    if args.bits:
        graph, books, walk_len = load_codebook(args.dataset)
        lang = Language(graph=graph, codebooks=books, walk_len=walk_len)
        bits = args.bits.strip()
        if any(ch not in "01" for ch in bits) or not bits:
            ap.error("--bits must be a non-empty string of 0/1")
        return 0 if show(lang, bits, oracle=args.oracle) else 1

    ds = load_dataset(args.dataset)
    split = {"train": ds.splits.train, "valid": ds.splits.valid, "test": ds.splits.test}[args.split]
    for sample in split[: args.sample]:
        show(ds.language, sample.bits, truth=sample.walk, oracle=args.oracle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
