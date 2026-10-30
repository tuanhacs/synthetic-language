#!/usr/bin/env python3
"""Server sweep: one semantic QA pool, nested train subsets, prefix/suffix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_config(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and yaml.safe_load(path.read_text()) != document:
        raise ValueError(f"existing configuration differs: {path}; choose another --output-root")
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def run(script: str, arguments: list[str], log: Path, dry_run: bool) -> None:
    command = [sys.executable, "-B", "-u", str(ROOT / script), *map(str, arguments)]
    print("\n>>> " + " ".join(command), flush=True)
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace") as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("pool", "all"), default="all")
    ap.add_argument("--code-types", nargs="+", choices=("prefix", "suffix"), default=["prefix", "suffix"])
    ap.add_argument("--pool-train-pairs", type=int, default=3_000_000)
    ap.add_argument("--train-min", type=int, default=600_000)
    ap.add_argument("--train-max", type=int, default=2_000_000)
    ap.add_argument("--train-step", type=int, default=200_000)
    ap.add_argument("--valid-query-pairs", type=int, default=100_000)
    ap.add_argument("--test-query-pairs", type=int, default=100_000)
    ap.add_argument("--qa-mode", choices=("iid", "heldout", "cross-region"), default="cross-region")
    ap.add_argument("--assignment", choices=("disjoint", "overlap"), default="overlap")
    ap.add_argument("--cot-mode", choices=("none", "segments"), default="segments")
    ap.add_argument("--grid-size", type=int, default=6)
    ap.add_argument("--query-min", type=int, default=2)
    ap.add_argument("--query-max", type=int, default=4)
    ap.add_argument("--segment-min", type=int, default=2)
    ap.add_argument("--segment-max", type=int, default=14)
    ap.add_argument("--region-overlap-rows", type=int, default=2)
    ap.add_argument("--heldout-pairs", default="3-16,7-21,10-35")
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--shared-codewords", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoding-seed", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--early-stopping-patience", type=int, default=0)
    ap.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--eval-samples", type=int, default=1000)
    ap.add_argument("--gen-batch-size", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--latent-pool", type=Path, help="optional existing shared latent pool")
    ap.add_argument("--dry-run", action="store_true", help="print commands without generating files or starting jobs")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = parser()
    args = ap.parse_args(argv)
    for name in ("pool_train_pairs", "train_min", "train_max", "train_step",
                 "valid_query_pairs", "test_query_pairs", "max_steps", "batch_size",
                 "eval_every", "checkpoint_every", "eval_samples", "gen_batch_size"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if not args.train_min <= args.train_max <= args.pool_train_pairs:
        ap.error("require train-min <= train-max <= pool-train-pairs")
    if (args.train_max - args.train_min) % args.train_step:
        ap.error("train-max must be reachable from train-min using train-step")
    if min(args.d_model, args.n_layers, args.n_heads) <= 0 or args.d_model % args.n_heads:
        ap.error("model dimensions must be positive and d-model divisible by n-heads")
    if args.lr <= 0 or args.temperature <= 0:
        ap.error("lr and temperature must be positive")
    if args.eval_batches < 1 or args.early_stopping_patience < 0:
        ap.error("validation intervals/batches must be positive and patience non-negative")
    if not 0 <= args.early_stopping_min_delta < float("inf"):
        ap.error("early stopping min-delta must be finite and non-negative")
    sizes = list(range(args.train_min, args.train_max + 1, args.train_step))
    # Always establish prefix as the assignment-slot reference, even for a
    # suffix-only worker. Independent workers can share a pre-created pool.
    pairs = []
    if args.qa_mode == "heldout":
        try:
            pairs = [list(map(int, pair.split("-"))) for pair in args.heldout_pairs.split(",")]
            if any(len(pair) != 2 for pair in pairs):
                raise ValueError()
        except ValueError:
            ap.error("--heldout-pairs must look like 3-16,7-21,10-35")
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key not in ("stage", "dry_run", "output_root", "code_types")}
    # Preserve settings identity for existing step-only sweeps with all new
    # options disabled/defaulted.
    for key, default in (("eval_batches", 20),
                         ("early_stopping_patience", 0), ("early_stopping_min_delta", 0.0)):
        if settings[key] == default:
            settings.pop(key)
    tag = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    out = args.output_root or ROOT / "model" / "outputs" / "path_qa_sweeps" / (
        f"{args.qa_mode}_{args.assignment}_q{args.query_min}-{args.query_max}_{args.cot_mode}_{tag}"
    )
    out = out.resolve()
    pool = args.latent_pool.resolve() if args.latent_pool else out / "latent_pool"
    print(f"Output: {out}\nLatent pool: {pool}\nTrain pair budgets: {sizes}")
    print(f"Runs: {len(sizes)} x {len(args.code_types)} = {len(sizes) * len(args.code_types)}")

    task = {
        "type": "path-qa", "cot_mode": args.cot_mode,
        "query_len": [args.query_min, args.query_max],
        "segment_len": [args.segment_min, args.segment_max], "path_trials": 1000,
        "split_mode": {"iid": "iid", "heldout": "held-out-pairs", "cross-region": "cross-region"}[args.qa_mode],
        "pair_direction": "unordered", "held_out_pairs": pairs,
        "region_overlap_rows": args.region_overlap_rows,
    }

    def data_config(kind: str) -> dict:
        language = {
            "graph": f"grid-{args.grid_size}x{args.grid_size}",
            "code": {"type": f"{kind}-free", "base_size": 3,
                     "depth_range": [1, 4], "power_x": 5},
            "k": [args.k_min, args.k_max],
            "assignment": "arbitrary-overlap" if args.assignment == "overlap" else "disjoint-random",
        }
        if args.assignment == "overlap":
            language["overlap"] = {"support_size": [2, 4], "shared_codewords": args.shared_codewords,
                                   "max_restarts": 100, "candidate_trials": 10_000}
        return {"language": language,
                "data": {"seed": args.seed, "context_len": "auto",
                         "pool_query_pairs": args.pool_train_pairs + args.valid_query_pairs + args.test_query_pairs},
                "task": task}

    # Validate before starting an expensive pool generation.
    sys.path.insert(0, str(ROOT / "data"))
    from synthdata.config import parse_config
    from synthdata.graphs import graph_from_spec
    from synthdata.qa_latent import semantic_config
    canonical = parse_config(data_config("prefix"))
    for kind in args.code_types:
        parse_config(data_config(kind))
    if not args.dry_run:
        write_config(out / "sweep_settings.yaml", settings)
        for kind in ("prefix", "suffix"):
            write_config(out / "configs" / f"data_{kind}.yaml", data_config(kind))
    pool_config = out / "configs" / "data_prefix.yaml"
    if not (pool / "manifest.json").exists():
        run("data/scripts/gen_qa_latent.py", [str(pool_config), "--out", str(pool),
            "--train-query-pairs", str(args.pool_train_pairs),
            "--valid-query-pairs", str(args.valid_query_pairs),
            "--test-query-pairs", str(args.test_query_pairs),
            "--checkpoints", *map(str, sorted(set(sizes + [args.pool_train_pairs])))],
            out / "logs" / "pool.log", args.dry_run)
    elif not args.dry_run:
        manifest = json.loads((pool / "manifest.json").read_text())
        if (manifest["semantic_config"] != semantic_config(graph_from_spec(canonical.language.graph), canonical.task)
                or manifest["seed"] != args.seed
                or manifest["requested_query_pairs"]["valid"] != args.valid_query_pairs
                or manifest["requested_query_pairs"]["test"] != args.test_query_pairs
                or manifest["stats"]["train"]["query_pair_occurrences"] < args.pool_train_pairs):
            raise ValueError("existing latent pool differs from the requested experiment")
        print("Reusing existing latent pool.")
    if args.stage == "pool":
        return 0

    template = out / "datasets" / "prefix" / str(sizes[0]) / "codebook.json"
    references = {}
    # Prepare prefix once if this worker trains only suffix.
    kinds = ["prefix", *[kind for kind in dict.fromkeys(args.code_types) if kind != "prefix"]]
    for kind in kinds:
        for count in sizes if kind in args.code_types else sizes[:1]:
            name = f"{kind}_{count}pairs"
            dataset = out / "datasets" / kind / str(count)
            source_config = out / "configs" / f"data_{kind}.yaml"
            generation_args = [str(source_config), "--out", str(dataset),
                               "--latent-pool", str(pool), "--train-query-pairs", str(count),
                               "--encoding-seed", str(args.encoding_seed), "--entropy-samples", "0"]
            if kind != "prefix":
                generation_args += ["--assignment-template", str(template)]
            if not (dataset / "manifest.json").exists():
                run("data/scripts/gen_dataset.py", generation_args,
                    out / "logs" / f"{name}_data.log", args.dry_run)
            if args.dry_run:
                context = 0  # context is measured after actual encoding
            else:
                manifest = json.loads((dataset / "manifest.json").read_text())
                provenance = manifest["stats"]["latent_pool"]
                pool_manifest = json.loads((pool / "manifest.json").read_text())
                if (provenance["source_sha256"] != pool_manifest["sha256"] or
                        provenance["encoding_seed"] != args.encoding_seed or
                        provenance["train_query_pairs_requested"] != count or
                        manifest["config"] != parse_config(data_config(kind)).to_dict()):
                    raise ValueError(f"existing encoded dataset differs: {dataset}")
                # Every size must use identical codebooks and fixed encoded eval splits.
                reference = tuple(digest(dataset / filename)
                                  for filename in ("codebook.json", "valid.jsonl", "test.jsonl"))
                if kind in references and references[kind] != reference:
                    raise ValueError(f"codebooks or fixed eval samples changed across {kind} train sizes")
                references[kind] = reference
                context = manifest["stats"]["recommended_context_len"]
            if kind not in args.code_types:
                continue
            run_dir = out / "runs" / name
            model_config = {
                "model": {"size": "base", "d_model": args.d_model, "n_layers": args.n_layers,
                          "n_heads": args.n_heads, "vocab_size": 6, "context_len": context},
                "data": {"dataset_dir": str(dataset), "mode": "frozen"},
                "train": {"max_steps": args.max_steps, "batch_size": args.batch_size,
                          "lr": args.lr, "warmup": min(100, args.max_steps), "log_every": 50,
                          "eval_every": args.eval_every, "checkpoint_every": args.checkpoint_every,
                          "seed": args.seed, "resume": "auto", "device": "cuda", "out_dir": str(run_dir)},
                "eval": {"n_samples": args.eval_samples, "gen_batch_size": args.gen_batch_size,
                         "temperature": args.temperature, "seed": args.seed, "max_len": context},
            }
            model_path = out / "configs" / f"model_{name}.yaml"
            for key, default in (("eval_batches", 20),
                                 ("early_stopping_patience", 0), ("early_stopping_min_delta", 0.0)):
                if getattr(args, key) != default:
                    model_config["train"][key] = getattr(args, key)
            if not args.dry_run:
                write_config(model_path, model_config)
            trained = run_dir / "training_complete.json"
            if not (trained.exists() and (run_dir / "best.pt").exists()):
                run("model/scripts/train.py", [str(model_path)], out / "logs" / f"{name}_train.log", args.dry_run)
                if not args.dry_run:
                    summary_path = run_dir / "train_summary.json"
                    completion = json.loads(summary_path.read_text()) if summary_path.exists() else {"max_steps": args.max_steps}
                    trained.write_text(json.dumps(completion) + "\n")
            else:
                print(f"Training already completed: {name}")
            evaluated = run_dir / "evaluation_complete.json"
            if not (evaluated.exists() and (run_dir / "eval_report.json").exists()):
                run("model/scripts/evaluate.py", [str(run_dir / "best.pt"), "--dataset", str(dataset),
                    "--device", "cuda", "--n-samples", str(args.eval_samples),
                    "--gen-batch-size", str(args.gen_batch_size), "--temperature", str(args.temperature)],
                    out / "logs" / f"{name}_eval.log", args.dry_run)
                if not args.dry_run:
                    evaluated.write_text(json.dumps({"temperature": args.temperature}) + "\n")
            if not args.dry_run:
                report = json.loads((run_dir / "eval_report.json").read_text())
                metrics = report["generation"]["path_qa"]
                row = {
                    "code_type": kind, "train_pairs_requested": count,
                    "train_pairs_actual": manifest["stats"]["splits"]["train"]["query_pair_occurrences"],
                    "qa_mode": args.qa_mode, "assignment": args.assignment,
                    "cot_mode": args.cot_mode, "query_len": [args.query_min, args.query_max],
                    "d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads,
                    "temperature": args.temperature, "n_samples": metrics["n_samples"],
                    "decodable_pct": metrics["answer_decodable_pct"],
                    "success_pct": metrics["semantic_success_pct"],
                    "report": str(run_dir / "eval_report.json"),
                }
                train_summary_path = run_dir / "train_summary.json"
                train_summary = json.loads(train_summary_path.read_text()) if train_summary_path.exists() else {}
                row.update({key: train_summary.get(key) for key in (
                    "steps", "max_steps", "stopped_early",
                    "stop_reason", "best_valid_bits_per_token",
                )})
                summary_path = out / "results.json"
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
                summary = [r for r in summary if (r["code_type"], r["train_pairs_requested"]) != (kind, count)]
                summary.append(row)
                summary.sort(key=lambda r: (r["code_type"], r["train_pairs_requested"]))
                summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                print(f"Result {name}: decodable={row['decodable_pct']:.1f}% success={row['success_pct']:.1f}%")
    print(f"\nSweep complete. Reports: {out / 'runs'}; per-stage logs: {out / 'logs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
