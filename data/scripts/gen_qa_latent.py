#!/usr/bin/env python3
"""Create a reusable semantic QA pool, without generating any codewords."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _HERE)]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthdata.config import load_config
from synthdata.graphs import graph_from_spec
from synthdata.qa_latent import create_latent_pool, frequency_report_lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="QA YAML; codebook and encoded-bit budgets are ignored")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-query-pairs", type=int, required=True)
    parser.add_argument("--valid-query-pairs", type=int, required=True)
    parser.add_argument("--test-query-pairs", type=int, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=[])
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    manifest = create_latent_pool(
        args.out, graph_from_spec(cfg.language.graph), cfg.task, cfg.data.seed,
        args.train_query_pairs, args.valid_query_pairs, args.test_query_pairs,
        args.checkpoints,
    )
    for split, stats in manifest["stats"].items():
        pairs, edges = stats["query_pairs_directed"], stats["graph_edges_undirected"]
        print(f"{split}: query length histogram={stats['histograms']['query_vertices']}")
        print(f"  directed query-pair coverage: {pairs['seen']}/{pairs['eligible']}")
        print(f"  undirected graph-edge coverage: {edges['seen']}/{edges['eligible']}")
        for line in frequency_report_lines(stats):
            print(line)
    print(f"Histogram, frequencies, coverage and checkpoint reports: {Path(args.out) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
