#!/usr/bin/env python3
"""Inspect a generated dataset.

    python scripts/inspect.py outputs/smoke

Prints split sizes, sentence/codeword length distributions, vertex frequencies
and verifies that the splits are disjoint at string level.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Put the package root on sys.path and drop this scripts/ directory from it,
# so that scripts/inspect.py cannot shadow the stdlib "inspect" module.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthdata.storage import load_dataset  # noqa: E402


def histogram_line(counter: Counter[int], width: int = 40) -> list[str]:
    if not counter:
        return []
    top = max(counter.values())
    lines = []
    for key in sorted(counter):
        bar = "#" * max(1, round(width * counter[key] / top))
        lines.append(f"    {key:5d} | {counter[key]:8d} {bar}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset directory")
    ap.add_argument("--top-vertices", type=int, default=10, help="how many vertices to list")
    ap.add_argument("--validate", type=int, default=0, help="re-decode N test sentences as a check")
    args = ap.parse_args(argv)

    ds = load_dataset(args.dataset)
    cfg, lang, splits = ds.config, ds.language, ds.splits

    print(f"dataset:   {Path(args.dataset).resolve()}")
    print(f"config:    hash={ds.manifest['config_hash']}  graph={cfg.language.graph} "
          f"code={cfg.language.code.type} k={cfg.language.k}")
    cert = ds.manifest["certification"]
    print(f"regime:    {cert['regime']}  unique_decoding={cert['unique_decoding']}")
    print(f"           code={cert['code']}")
    print(f"           codebooks: theorem_d1={cert['codebooks']['theorem_d1']} "
          f"pairwise_disjoint={cert['codebooks']['pairwise_disjoint']}")
    floor = ds.manifest["stats"].get("entropy_floor_bits_per_token")
    if floor is not None:
        print(f"floor:     {floor:.4f} bits/token")

    pool_code = lang.codebooks.global_code()
    print(f"\ncodewords: {len(pool_code)} used ({cfg.language.num_vertices} vertices x {lang.codebooks.k})"
          f"  lengths {pool_code.min_len}..{pool_code.max_len}")
    print("  codeword length histogram (used codewords):")
    for line in histogram_line(Counter({length: n for length, n in pool_code.length_histogram().items()})):
        print(line)

    print("\nsplits:")
    all_bits: dict[str, set[str]] = {}
    for name, split in splits.items():
        all_bits[name] = {s.bits for s in split}
        if not split:
            print(f"  {name:5s} empty")
            continue
        lengths = [len(s.bits) for s in split]
        walks = [len(s.walk) for s in split]
        noised = sum(1 for s in split if s.noised_bits is not None)
        print(
            f"  {name:5s} {len(split):7d} sentences  {sum(lengths):9d} bits  "
            f"sentence {min(lengths)}..{max(lengths)} (mean {sum(lengths)/len(lengths):.1f})  "
            f"walk {min(walks)}..{max(walks)} (mean {sum(walks)/len(walks):.1f})  "
            f"noised {noised}"
        )

    print("\nsplit disjointness (string level):")
    names = list(all_bits)
    ok = True
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = all_bits[a] & all_bits[b]
            ok = ok and not shared
            print(f"  {a} n {b}: {len(shared)} shared" + ("" if not shared else "  <-- LEAK"))
    total = sum(len(v) for v in all_bits.values())
    union = len(set().union(*all_bits.values())) if all_bits else 0
    print(f"  unique across all splits: {union} / {total}")
    print(f"  => {'OK' if ok and union == total else 'FAILED'}")

    print("\nsentence length histogram (train, bucketed by 16 bits):")
    buckets = Counter(len(s.bits) // 16 * 16 for s in splits.train)
    for line in histogram_line(buckets):
        print(line)

    print("\nvertex frequencies (train walks):")
    vertex_counts = Counter(v for s in splits.train for v in s.walk)
    total_steps = sum(vertex_counts.values()) or 1
    for v, c in vertex_counts.most_common(args.top_vertices):
        print(f"    v{v:<4d} {c:9d}  {100 * c / total_steps:5.2f}%  deg={lang.graph.degree(v)}")
    missing = [v for v in lang.graph.vertices if v not in vertex_counts]
    print(f"    vertices never visited: {missing if missing else 'none'}")

    if args.validate:
        print(f"\nre-decoding {args.validate} test sentences:")
        exact = other = invalid = 0
        for sample in splits.test[: args.validate]:
            walks = lang.decode(sample.bits)
            if not walks:
                invalid += 1
            elif len(walks) == 1 and tuple(walks[0]) == sample.walk:
                exact += 1
            else:
                other += 1
        print(f"    unique & correct: {exact}  ambiguous/wrong: {other}  invalid: {invalid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
