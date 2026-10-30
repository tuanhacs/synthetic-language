"""Optional W&B tracking. Local JSON metrics/checkpoints remain authoritative."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

from config import Config


def flatten_metrics(document: dict, prefix: str) -> dict:
    result = {}
    for name, value in document.items():
        key = f"{prefix}/{name}"
        if isinstance(value, dict):
            result.update(flatten_metrics(value, key))
        elif isinstance(value, (str, bool, int)) or isinstance(value, float) and math.isfinite(value):
            result[key] = value
    return result


class RunTracker:
    def __init__(self, cfg: Config, out_dir: Path | None = None, resume: bool = True):
        self.run = None
        self.mode = cfg.wandb.mode
        if self.mode == "disabled":
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B is enabled; install it with: python -m pip install wandb") from exc
        directory = out_dir or cfg.resolved_out_dir()
        directory.mkdir(parents=True, exist_ok=True)
        identity_path = directory / "wandb_run.json"
        identity = json.loads(identity_path.read_text()) if resume and identity_path.exists() else None
        if identity and (identity["project"], identity["entity"], identity["mode"]) != (
            cfg.wandb.project, cfg.wandb.entity, self.mode
        ):
            raise ValueError("W&B project/entity/mode differs from wandb_run.json; use a new output directory")
        if identity is None:
            identity = {"id": uuid.uuid4().hex[:12], "project": cfg.wandb.project,
                        "entity": cfg.wandb.entity, "mode": self.mode}
        kwargs = {
            **identity, "name": cfg.wandb.name or directory.name,
            "group": cfg.wandb.group, "tags": list(cfg.wandb.tags),
            "dir": str(directory), "config": cfg.to_dict(),
            "job_type": "train-evaluate", "allow_val_change": True,
        }
        # W&B only supports cloud history resume in online mode. Offline
        # restarts write new local segments, which can be synced separately.
        if self.mode == "online":
            kwargs["resume"] = "allow"
        self.run = wandb.init(**kwargs)
        temporary = identity_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        temporary.replace(identity_path)
        self.run.define_metric("trainer/step")
        self.run.define_metric("*", step_metric="trainer/step")

    def log_training(self, record: dict) -> None:
        if self.run is None:
            return
        values = {
            "trainer/step": record["step"], "train/lr": record["lr"],
            "train/loss_bits_per_token": record["train_bits_per_token"],
            "train/elapsed_s": record["elapsed_s"],
            "early_stopping/bad_checks": record["early_stopping_bad_checks"],
        }
        if record["valid_bits_per_token"] is not None:
            values["valid/loss_bits_per_token"] = record["valid_bits_per_token"]
        self.run.log(values)

    def summary(self, values: dict, prefix: str = "train") -> None:
        if self.run is not None:
            self.run.summary.update(flatten_metrics(values, prefix))

    def dataset(self, manifest: dict, num_params: int, num_windows: int | None) -> None:
        if self.run is not None:
            stats = manifest.get("stats", {})
            language = manifest.get("config", {}).get("language", {})
            self.run.config.update({
                "code_type": language.get("code", {}).get("type"),
                "assignment": language.get("assignment"),
                "qa_mode": manifest.get("config", {}).get("task", {}).get("split_mode"),
                "train_query_pairs": stats.get("splits", {}).get("train", {}).get("query_pair_occurrences"),
                "dataset": {
                    "config_hash": manifest.get("config_hash"),
                    "fingerprint": manifest.get("dataset_fingerprint"),
                    "language": manifest.get("config", {}).get("language"),
                    "task": manifest.get("config", {}).get("task"),
                    "splits": {name: {key: value for key, value in report.items()
                                      if key in ("num_sentences", "total_bits", "query_pair_occurrences")}
                               for name, report in stats.get("splits", {}).items()},
                    "num_train_windows": num_windows,
                }, "num_params": num_params,
            }, allow_val_change=True)

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            self.run.finish(exit_code=exit_code)


def log_evaluation(cfg: Config, checkpoint: Path, report: dict, step: int) -> None:
    tracker = RunTracker(cfg, out_dir=checkpoint.resolve().parent)
    try:
        generation = report.get("generation", {})
        values = flatten_metrics(generation.get("path_qa", generation), "test")
        if "path_qa" in generation:
            values["test/decodable_pct"] = generation["path_qa"]["answer_decodable_pct"]
            values["test/success_pct"] = generation["path_qa"]["semantic_success_pct"]
        if tracker.run is not None:
            tracker.run.log({"trainer/step": step, **values})
            tracker.run.summary.update(values)
        tracker.finish()
    except BaseException:
        tracker.finish(exit_code=1)
        raise
