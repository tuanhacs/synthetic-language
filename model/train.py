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
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import Config
from data import (
    BatchSampler,
    IGNORE_INDEX,
    PackedData,
    load_frozen,
    resolve_device,
    seed_everything,
    sentence_batches,
    stream_batches,
)
from evals import LN2, bits_per_token
from model import build_model


def lr_at(step: int, cfg) -> float:
    """Linear warmup then cosine decay to ``min_lr_frac`` of the peak."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.max_steps - cfg.warmup)
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
    """Train according to ``cfg``; returns a summary dict.

    Writes ``metrics.jsonl``, ``best.pt`` and ``last.pt`` into ``train.out_dir``.
    A checkpoint holds ``{model: state_dict, config: full config dict, step, ...}``.
    """
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    out_dir = cfg.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    data: PackedData = load_frozen(cfg.resolved_dataset_dir(), cfg.model.context_len)
    floor = data.entropy_floor
    model = build_model(cfg.model, data.tokenizer).to(device)
    optimizer = make_optimizer(model, cfg.train)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

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
        sampler = BatchSampler(data.train, cfg.train.batch_size, seed=cfg.train.seed)
        next_train_batch = lambda: sampler.batch(device)  # noqa: E731
        n_train_windows = len(sampler)

    valid_sampler = (
        BatchSampler(data.valid, cfg.train.batch_size, seed=cfg.train.seed + 1)
        if data.valid.numel()
        else None
    )

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

    best_valid = float("inf")
    history: list[dict] = []
    start_time = time.time()
    first_train_bits: float | None = None
    last_train_bits = float("nan")

    for step in range(cfg.train.max_steps):
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
        do_eval = valid_sampler is not None and (
            (step + 1) % cfg.train.eval_every == 0 or is_last or step == 0
        )
        valid_bits = None
        if do_eval:
            valid_bits = bits_per_token(
                model,
                valid_sampler.sequential_batches(cfg.train.eval_batches, device),
                device,
            )
            if valid_bits < best_valid:
                best_valid = valid_bits
                save_checkpoint(out_dir / "best.pt", model, cfg, step + 1, valid_bits)

        if (step + 1) % cfg.train.log_every == 0 or is_last or step == 0 or do_eval:
            record = {
                "step": step + 1,
                "lr": lr,
                "train_bits_per_token": train_bits,
                "valid_bits_per_token": valid_bits,
                "gap_to_floor": None if (floor is None or valid_bits is None) else valid_bits - floor,
                "elapsed_s": time.time() - start_time,
            }
            history.append(record)
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            if verbose:
                parts = [f"step {step + 1:>6}/{cfg.train.max_steps}", f"lr {lr:.2e}",
                         f"train {train_bits:.4f} b/t"]
                if valid_bits is not None:
                    parts.append(f"valid {valid_bits:.4f} b/t")
                    if floor is not None:
                        parts.append(f"gap {valid_bits - floor:+.4f}")
                parts.append(f"{record['elapsed_s']:.0f}s")
                print("  ".join(parts))

    save_checkpoint(out_dir / "last.pt", model, cfg, cfg.train.max_steps, best_valid)

    # Final sentence-aligned valid loss: the quantity directly comparable to the floor.
    final_valid = (
        bits_per_token(
            model, sentence_batches(data.valid_samples, cfg.train.batch_size, data.tokenizer), device
        )
        if data.valid_samples
        else None
    )
    summary = {
        "out_dir": str(out_dir),
        "device": str(device),
        "num_params": model.num_params(),
        "steps": cfg.train.max_steps,
        "first_train_bits_per_token": first_train_bits,
        "last_train_bits_per_token": last_train_bits,
        "best_valid_bits_per_token": None if best_valid == float("inf") else best_valid,
        "final_valid_bits_per_token_sentence_aligned": final_valid,
        "entropy_floor_bits_per_token": floor,
        "gap_to_floor": None if (floor is None or final_valid is None) else final_valid - floor,
        "elapsed_s": time.time() - start_time,
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2) + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(
            f"done in {summary['elapsed_s']:.0f}s  best valid {summary['best_valid_bits_per_token']:.4f} b/t"
            f"  final valid (sentence-aligned) {final_valid:.4f} b/t"
            + (f"  gap {summary['gap_to_floor']:+.4f}" if summary["gap_to_floor"] is not None else "")
        )
    return summary


def save_checkpoint(path: Path, model, cfg: Config, step: int, valid_bits: float) -> None:
    """Checkpoint = state_dict + the full run config + step."""
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "step": step,
            "valid_bits_per_token": valid_bits,
        },
        path,
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
