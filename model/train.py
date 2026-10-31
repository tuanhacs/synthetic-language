"""Training loop (library function; ``scripts/train.py`` is the CLI).

AdamW with decoupled weight decay (no decay on norms and the tied embedding),
linear warmup then cosine decay to ``min_lr_frac * lr``, gradient clipping at 1.0.
Loss is reported in **bits/token** (nats / ln 2) so it is directly comparable with
the dataset manifest's entropy floor.

Device: cuda > mps > cpu. AMP (bf16) is used on cuda only; mps and cpu run fp32 —
deliberately simple and robust rather than fast.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from config import Config
from data import (
    BatchSampler,
    IGNORE_INDEX,
    PackedData,
    load_frozen_train_valid,
    resolve_device,
    seed_everything,
    stream_batches,
)
from evals import LN2, bits_per_token
from model import build_model
from tracking import RunTracker


@dataclass
class EarlyStopping:
    best: float = float("inf")
    bad_checks: int = 0
    stopped: bool = False

    def update(self, value: float, cfg) -> bool:
        if not math.isfinite(value):
            raise ValueError("validation loss must be finite for early stopping")
        if value < self.best - cfg.early_stopping_min_delta:
            self.best = value
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        self.stopped = bool(cfg.early_stopping_patience and
                            self.bad_checks >= cfg.early_stopping_patience)
        return self.stopped


def lr_at(step: int, cfg) -> float:
    """Warm up, decay to the floor at the configured step, then hold it."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    decay_steps = cfg.lr_decay_steps if cfg.lr_decay_steps is not None else cfg.max_steps
    progress = (step - cfg.warmup) / max(1, decay_steps - cfg.warmup)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_frac + (1.0 - cfg.min_lr_frac) * cosine)


def make_optimizer(model, cfg) -> torch.optim.AdamW:
    """AdamW with weight decay only on 2D matrices (no norms, no embedding)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and "embed" not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=tuple(cfg.betas),
    )


def train(cfg: Config, verbose: bool = True) -> dict:
    """Run step-based training with optional W&B metrics and failure cleanup."""
    tracker = RunTracker(cfg, resume=cfg.train.resume != "never")
    try:
        result = _train(cfg, verbose, tracker)
        tracker.summary({key: value for key, value in result.items() if key != "history"})
        tracker.finish()
        return result
    except BaseException:
        tracker.finish(exit_code=1)
        raise


def _train(cfg: Config, verbose: bool, tracker: RunTracker) -> dict:
    """Train according to ``cfg``; returns a summary dict.

    Writes ``metrics.jsonl``, validation-selected ``best.pt`` and resumable
    ``latest.pt`` into ``train.out_dir``.
    """
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    out_dir = cfg.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    data: PackedData = load_frozen_train_valid(
        cfg.resolved_dataset_dir(), cfg.model.context_len
    )
    task_type = data.manifest.get("config", {}).get("task", {}).get(
        "type", "language-modeling"
    )
    if cfg.data.mode == "streaming" and task_type == "path-qa":
        raise ValueError("path-QA currently requires data.mode: frozen")
    floor = data.entropy_floor
    model = build_model(cfg.model, data.tokenizer).to(device)
    optimizer = make_optimizer(model, cfg.train)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    sampler: BatchSampler | None = None
    if cfg.data.mode == "streaming":
        from synthdata.config import parse_config as parse_data_config

        data_cfg = parse_data_config(data.manifest["config"]).data
        seed = cfg.data.stream_seed if cfg.data.stream_seed is not None else cfg.train.seed
        train_iter = stream_batches(
            data.language, data_cfg, cfg.model.context_len, cfg.train.batch_size, seed, device
        )
        next_train_batch = lambda: next(train_iter)  # noqa: E731
        n_train_windows = None
    else:
        sampler = BatchSampler(
            data.train, cfg.train.batch_size, seed=cfg.train.seed, tokenizer=data.tokenizer
        )
        next_train_batch = lambda: sampler.batch(device)  # noqa: E731
        n_train_windows = len(sampler)
    tracker.dataset(data.manifest, model.num_params(), n_train_windows)

    valid_sampler = (
        BatchSampler(
            data.valid,
            cfg.train.batch_size,
            seed=cfg.train.seed + 1,
            tokenizer=data.tokenizer,
        )
        if data.valid.numel()
        else None
    )
    if cfg.train.early_stopping_patience and valid_sampler is None:
        raise ValueError("early stopping requires a non-empty validation split")

    latest_path = out_dir / "latest.pt"
    should_resume = cfg.train.resume in ("auto", "require") and latest_path.exists()
    if cfg.train.resume == "require" and not latest_path.exists():
        raise FileNotFoundError(f"resume required but checkpoint does not exist: {latest_path}")

    start_step = 0
    best_valid = float("inf")
    early_stopping = EarlyStopping()
    if should_resume:
        # RNG states must remain CPU ByteTensors. Loading the whole checkpoint
        # directly onto CUDA turns them into CUDA tensors, which
        # torch.Generator.set_state()/torch.set_rng_state() reject. Model and
        # optimizer load_state_dict calls below move their tensors as needed.
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        _validate_resume_checkpoint(checkpoint, cfg, data)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_valid = float(checkpoint.get("best_valid_bits_per_token", float("inf")))
        early_stopping = EarlyStopping(**checkpoint.get("early_stopping", {}))
        if sampler is None:
            raise ValueError(
                "exact checkpoint resume is currently supported only for frozen data"
            )
        sampler_state = checkpoint.get("train_sampler")
        if sampler_state is None:
            raise ValueError("latest.pt has no frozen-data sampler state")
        sampler.load_state_dict(sampler_state)
        _restore_rng_state(checkpoint["rng_state"])
        _trim_metrics(metrics_path, start_step)
    else:
        metrics_path.write_text("", encoding="utf-8")

    if verbose:
        if cfg.train.warmup >= cfg.train.max_steps:
            print(f"warning: warmup {cfg.train.warmup} >= max_steps {cfg.train.max_steps} "
                  "— the run never leaves the warmup ramp")
        print(f"device        {device}")
        print(f"model         {cfg.model.size or 'custom'}  {model.num_params():,} params "
              f"({model.num_params(non_embedding=True):,} non-embedding)")
        print(f"data          {data.dataset_dir}  mode={cfg.data.mode}")
        print(f"              train windows: {n_train_windows}  tokens: {data.token_counts()}")
        print(f"entropy floor {floor if floor is None else f'{floor:.4f}'} bits/token")
        print(f"out_dir       {out_dir}")
        if cfg.train.early_stopping_patience:
            print(f"early stop    patience={cfg.train.early_stopping_patience} validation checks, "
                  f"min_delta={cfg.train.early_stopping_min_delta} b/t")
        if should_resume:
            print(f"resume        {latest_path}  (completed step {start_step})")
        else:
            print(f"resume        no ({cfg.train.resume})")

    if start_step >= cfg.train.max_steps or early_stopping.stopped:
        if verbose:
            print(
                f"training already complete at step {start_step}; "
                "leaving checkpoints unchanged"
            )
        completed = {
            "out_dir": str(out_dir),
            "device": str(device),
            "num_params": model.num_params(),
            "steps": start_step,
            "best_valid_bits_per_token": (
                None if best_valid == float("inf") else best_valid
            ),
            "resumed_complete": True,
            "history": [],
            "max_steps": cfg.train.max_steps,
            "stopped_early": early_stopping.stopped,
            "stop_reason": "early_stopping" if early_stopping.stopped else "max_steps",
        }
        summary_path = out_dir / "train_summary.json"
        if not summary_path.exists():
            summary_path.write_text(json.dumps({k: v for k, v in completed.items() if k != "history"}, indent=2) + "\n",
                                    encoding="utf-8")
        return completed

    history: list[dict] = []
    start_time = time.time()
    first_train_bits: float | None = None
    last_train_bits = float("nan")
    last_valid_bits: float | None = None
    completed_steps = start_step

    for step in range(start_step, cfg.train.max_steps):
        lr = lr_at(step, cfg.train)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        inputs, targets = next_train_batch()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    targets.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
        else:
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        train_bits = float(loss.detach()) / LN2
        last_train_bits = train_bits
        if first_train_bits is None:
            first_train_bits = train_bits

        is_last = step + 1 == cfg.train.max_steps
        completed_steps = step + 1
        do_eval = valid_sampler is not None and (
            completed_steps % cfg.train.eval_every == 0 or is_last or step == 0
        )
        valid_bits = None
        if do_eval:
            valid_bits = bits_per_token(
                model,
                valid_sampler.sequential_batches(cfg.train.eval_batches, device),
                device,
            )
            last_valid_bits = valid_bits
            if valid_bits < best_valid:
                best_valid = valid_bits
                save_checkpoint(out_dir / "best.pt", model, cfg, step + 1, valid_bits)
            if cfg.train.early_stopping_patience:
                early_stopping.update(valid_bits, cfg.train)

        if (step + 1) % cfg.train.log_every == 0 or is_last or step == 0 or do_eval:
            record = {
                "step": step + 1,
                "lr": lr,
                "train_bits_per_token": train_bits,
                "valid_bits_per_token": valid_bits,
                "elapsed_s": time.time() - start_time,
                "early_stopping_bad_checks": early_stopping.bad_checks,
            }
            history.append(record)
            tracker.log_training(record)
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            if verbose:
                parts = [f"step {step + 1:>6}/{cfg.train.max_steps}", f"lr {lr:.2e}",
                         f"train {train_bits:.4f} b/t"]
                if valid_bits is not None:
                    parts.append(f"valid {valid_bits:.4f} b/t")
                if do_eval and cfg.train.early_stopping_patience:
                    parts.append(f"patience {early_stopping.bad_checks}/{cfg.train.early_stopping_patience}")
                parts.append(f"{record['elapsed_s']:.0f}s")
                print("  ".join(parts))

        if (step + 1) % cfg.train.checkpoint_every == 0 or is_last or early_stopping.stopped:
            save_training_checkpoint(
                latest_path,
                model,
                optimizer,
                cfg,
                data,
                step + 1,
                best_valid,
                sampler,
                early_stopping_state=asdict(early_stopping),
            )
            if verbose:
                print(f"checkpoint    {latest_path}  (step {step + 1})", flush=True)
        if early_stopping.stopped:
            if verbose:
                print(f"early stopping at step {completed_steps}; "
                      f"no significant validation improvement for {early_stopping.bad_checks} checks", flush=True)
            break

    summary = {
        "out_dir": str(out_dir),
        "device": str(device),
        "num_params": model.num_params(),
        "steps": completed_steps,
        "max_steps": cfg.train.max_steps,
        "stopped_early": early_stopping.stopped,
        "stop_reason": "early_stopping" if early_stopping.stopped else "max_steps",
        "early_stopping_bad_checks": early_stopping.bad_checks,
        "first_train_bits_per_token": first_train_bits,
        "last_train_bits_per_token": last_train_bits,
        "best_valid_bits_per_token": None if best_valid == float("inf") else best_valid,
        "final_valid_bits_per_token": last_valid_bits,
        "entropy_floor_bits_per_token": floor,
        "elapsed_s": time.time() - start_time,
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2) + "\n",
        encoding="utf-8",
    )
    if verbose:
        parts = [
            f"done in {summary['elapsed_s']:.0f}s",
            f"best valid {summary['best_valid_bits_per_token']} b/t",
        ]
        if last_valid_bits is not None:
            parts.append(
                f"final valid ({cfg.train.eval_batches} batches) "
                f"{last_valid_bits:.4f} b/t"
            )
        print("  ".join(parts))
    return summary


def save_checkpoint(path: Path, model, cfg: Config, step: int, valid_bits: float) -> None:
    """Checkpoint = state_dict + the full run config + step."""
    _atomic_torch_save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "step": step,
            "valid_bits_per_token": valid_bits,
        },
        path,
    )


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Replace a checkpoint only after its temporary file was written fully."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(_as_cpu_rng_state(state["torch_cpu"]))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [_as_cpu_rng_state(item) for item in state["torch_cuda"]]
        )


def _as_cpu_rng_state(state) -> torch.Tensor:
    """Normalize legacy/device-mapped RNG state for PyTorch RNG APIs."""
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state, dtype=torch.uint8)
    return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _trim_metrics(path: Path, completed_step: int) -> None:
    """Drop log records newer than the durable training checkpoint."""
    if not path.exists():
        path.touch()
        return
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("step", 0)) <= completed_step:
            kept.append(json.dumps(record))
    path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def _resume_signature(cfg: Config) -> dict:
    """Fields that must stay fixed for an exact continuation."""
    train = cfg.train
    signature = {
        "model": cfg.model.to_dict(),
        "data_mode": cfg.data.mode,
        "train": {
            key: getattr(train, key)
            for key in (
                "max_steps",
                "batch_size",
                "lr",
                "min_lr_frac",
                "warmup",
                "weight_decay",
                "betas",
                "grad_clip",
                "seed",
            )
        },
    }
    # Omit the default field so legacy checkpoints keep their exact signature.
    if train.lr_decay_steps is not None:
        signature["train"]["lr_decay_steps"] = train.lr_decay_steps
    if train.early_stopping_patience:
        signature["stopping"] = {
            key: getattr(train, key) for key in (
                "eval_every", "eval_batches",
                "early_stopping_patience", "early_stopping_min_delta",
            )
        }
    return signature


def save_training_checkpoint(
    path: Path,
    model,
    optimizer,
    cfg: Config,
    data: PackedData,
    step: int,
    best_valid: float,
    sampler: BatchSampler | None,
    early_stopping_state: dict | None = None,
) -> None:
    """Write the complete state required to continue training."""
    _atomic_torch_save(
        {
            "checkpoint_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "resume_signature": _resume_signature(cfg),
            "dataset_config_hash": data.manifest.get("dataset_fingerprint", data.manifest.get("config_hash")),
            "step": step,
            "best_valid_bits_per_token": best_valid,
            "train_sampler": None if sampler is None else sampler.state_dict(),
            "early_stopping": early_stopping_state or asdict(EarlyStopping()),
            "rng_state": _rng_state(),
        },
        path,
    )


def _validate_resume_checkpoint(checkpoint: dict, cfg: Config, data: PackedData) -> None:
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("latest.pt is not a supported resumable checkpoint")
    if checkpoint.get("resume_signature") != _resume_signature(cfg):
        raise ValueError(
            "latest.pt is incompatible with the current model/training config; "
            "use train.resume: never or a different out_dir"
        )
    expected_hash = data.manifest.get("dataset_fingerprint", data.manifest.get("config_hash"))
    if checkpoint.get("dataset_config_hash") != expected_hash:
        raise ValueError(
            "latest.pt belongs to a different dataset config hash: "
            f"{checkpoint.get('dataset_config_hash')} != {expected_hash}"
        )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    """Rebuild ``(model, Config, checkpoint dict)`` from a checkpoint file."""
    from config import parse_config

    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    raw = dict(ckpt["config"])
    path_hint = raw.pop("path", None)
    cfg = parse_config(raw, path=path_hint)
    model = build_model(cfg.model)
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg, ckpt
