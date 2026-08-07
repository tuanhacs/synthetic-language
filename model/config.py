"""Configuration schema for model / training / evaluation.

One run = one YAML file. The model family is a single architecture scaled purely
by :data:`SIZE_PRESETS`, so a size sweep is a sweep over one string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from _paths import MODEL_ROOT  # the ``model/`` directory; relative config paths hang off it

#: Named model family for scaling-law sweeps: size -> (d_model, n_layers, n_heads).
#: With a 5-token vocabulary the embedding is negligible, so parameter counts are
#: essentially all non-embedding.
SIZE_PRESETS: dict[str, dict[str, int]] = {
    "nano": {"d_model": 64, "n_layers": 2, "n_heads": 2},
    "micro": {"d_model": 128, "n_layers": 4, "n_heads": 4},
    "tiny": {"d_model": 256, "n_layers": 6, "n_heads": 4},
    "small": {"d_model": 384, "n_layers": 8, "n_heads": 6},
    "base": {"d_model": 512, "n_layers": 10, "n_heads": 8},
    "large": {"d_model": 768, "n_layers": 12, "n_heads": 12},
}


class ConfigError(ValueError):
    """Raised when a run config is malformed."""


@dataclass(frozen=True)
class ModelConfig:
    """Architecture. ``size`` is bookkeeping only; the four numbers below rule."""

    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    context_len: int = 512
    vocab_size: int = 5  # {0, 1, BOS, EOS, PAD} — asserted against BitTokenizer
    dropout: float = 0.0
    size: str | None = None

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ConfigError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        for name in ("d_model", "n_layers", "n_heads", "context_len", "vocab_size"):
            if getattr(self, name) < 1:
                raise ConfigError(f"model.{name} must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        """SwiGLU hidden width: 4 * d_model (gate style, so 3 matrices of this width)."""
        return 4 * self.d_model

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "context_len": self.context_len,
            "vocab_size": self.vocab_size,
            "dropout": self.dropout,
        }


@dataclass(frozen=True)
class DataConfig:
    """Which dataset, and how it is fed."""

    dataset_dir: str = "../data/outputs/smoke"
    mode: str = "frozen"  # frozen | streaming
    stream_seed: int | None = None  # streaming only; defaults to the train seed

    def to_dict(self) -> dict[str, Any]:
        return {"dataset_dir": self.dataset_dir, "mode": self.mode, "stream_seed": self.stream_seed}


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 2000
    batch_size: int = 32
    lr: float = 3e-4
    min_lr_frac: float = 0.1  # cosine floor, as a fraction of lr
    warmup: int = 100
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 250
    eval_batches: int = 20  # batches per validation pass
    seed: int = 42
    out_dir: str = "outputs/run"
    device: str | None = None  # None = auto (cuda > mps > cpu)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["betas"] = list(self.betas)
        return d


@dataclass(frozen=True)
class EvalConfig:
    n_samples: int = 200
    temperature: float = 1.0
    cuts: tuple[int, ...] = (0, 50)
    max_len: int | None = None  # generation cap in tokens; None = model context_len
    gen_batch_size: int = 64
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "temperature": self.temperature,
            "cuts": list(self.cuts),
            "max_len": self.max_len,
            "gen_batch_size": self.gen_batch_size,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    path: str | None = None  # config file this came from, if any

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "train": self.train.to_dict(),
            "eval": self.eval.to_dict(),
            "path": self.path,
        }

    def resolved_dataset_dir(self) -> Path:
        """``data.dataset_dir``, resolved against the ``model/`` root (see :func:`_resolve`)."""
        return _resolve(self.data.dataset_dir)

    def resolved_out_dir(self) -> Path:
        """``train.out_dir``, resolved against the ``model/`` root (see :func:`_resolve`)."""
        return _resolve(self.train.out_dir)


def _resolve(raw_path: str) -> Path:
    """Absolute path: as given if absolute, else relative to :data:`MODEL_ROOT`."""
    raw = Path(raw_path)
    return raw if raw.is_absolute() else (MODEL_ROOT / raw).resolve()


def _check_keys(raw: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown {where} fields: {sorted(unknown)}")


def _model_config(raw: dict[str, Any]) -> ModelConfig:
    _check_keys(
        raw,
        {"size", "d_model", "n_layers", "n_heads", "context_len", "vocab_size", "dropout"},
        "model",
    )
    size = raw.get("size")
    fields: dict[str, Any] = {}
    if size is not None:
        if size not in SIZE_PRESETS:
            raise ConfigError(f"model.size must be one of {sorted(SIZE_PRESETS)}, got {size!r}")
        fields.update(SIZE_PRESETS[size])
    for key in ("d_model", "n_layers", "n_heads", "context_len", "vocab_size"):
        if key in raw:
            fields[key] = int(raw[key])  # explicit values override the preset
    if "dropout" in raw:
        fields["dropout"] = float(raw["dropout"])
    return ModelConfig(size=size, **fields)


def parse_config(raw: dict[str, Any], path: str | None = None) -> Config:
    """Build a validated :class:`Config` from a plain dict."""
    _check_keys(raw, {"model", "data", "train", "eval"}, "top-level")

    data_raw = dict(raw.get("data") or {})
    _check_keys(data_raw, {"dataset_dir", "mode", "stream_seed"}, "data")
    mode = str(data_raw.get("mode", "frozen"))
    if mode not in ("frozen", "streaming"):
        raise ConfigError(f"data.mode must be 'frozen' or 'streaming', got {mode!r}")

    train_raw = dict(raw.get("train") or {})
    _check_keys(train_raw, set(TrainConfig().__dict__), "train")
    if "betas" in train_raw:
        b = train_raw["betas"]
        if not isinstance(b, (list, tuple)) or len(b) != 2:
            raise ConfigError("train.betas must be a [beta1, beta2] pair")
        train_raw["betas"] = (float(b[0]), float(b[1]))

    eval_raw = dict(raw.get("eval") or {})
    _check_keys(eval_raw, set(EvalConfig().to_dict()), "eval")
    if "cuts" in eval_raw:
        cuts = eval_raw["cuts"]
        if not isinstance(cuts, (list, tuple)) or any(int(c) < 0 for c in cuts):
            raise ConfigError("eval.cuts must be a list of non-negative prefix lengths")
        eval_raw["cuts"] = tuple(int(c) for c in cuts)

    cfg = Config(
        model=_model_config(dict(raw.get("model") or {})),
        data=DataConfig(
            dataset_dir=str(data_raw.get("dataset_dir", DataConfig.dataset_dir)),
            mode=mode,
            stream_seed=(
                None if data_raw.get("stream_seed") is None else int(data_raw["stream_seed"])
            ),
        ),
        train=TrainConfig(**train_raw),
        eval=EvalConfig(**eval_raw),
        path=path,
    )
    return cfg


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML run config."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return parse_config(raw, path=str(path))
