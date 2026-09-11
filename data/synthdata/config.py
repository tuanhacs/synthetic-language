"""Configuration schema, loading and validation.

One dataset = one YAML config + a seed. Any change inside the ``language`` block
defines a *different language*, hence a different dataset.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CODE_TYPES = ("prefix-free", "suffix-free", "ud")
ASSIGNMENTS = ("disjoint-random", "arbitrary-overlap")
NOISE_TYPES = ("bit-flip", "bit-delete", "vertex-noise")
TASK_TYPES = ("language-modeling", "path-qa")
QA_SPLIT_MODES = ("iid", "held-out-pairs", "cross-region")
PAIR_DIRECTIONS = ("ordered", "unordered")

_GRAPH_RE = re.compile(r"^grid[-_ ]?(\d+)\s*x\s*(\d+)$", re.IGNORECASE)


class ConfigError(ValueError):
    """Raised when a config is malformed or violates a cross-constraint."""


@dataclass(frozen=True)
class CodeConfig:
    """Base-code generation parameters.

    ``depth_range`` is used by the tree-pruning generators (prefix-free /
    suffix-free); ``len_range`` is used by the UD rejection sampler. Both are
    accepted under either key in YAML and normalised here into
    :attr:`length_range`.
    """

    type: str = "prefix-free"
    base_size: int = 3
    depth_range: tuple[int, int] | None = None
    len_range: tuple[int, int] | None = None
    power_x: int = 6
    max_tries: int = 10_000

    @property
    def length_range(self) -> tuple[int, int]:
        """Codeword length range of the *base* code, whichever key was given."""
        rng = self.len_range if self.type == "ud" else self.depth_range
        if rng is None:  # fall back to the other key
            rng = self.depth_range if self.type == "ud" else self.len_range
        if rng is None:
            raise ConfigError("code: depth_range (or len_range) is required")
        return rng

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.type,
            "base_size": self.base_size,
            "power_x": self.power_x,
            "max_tries": self.max_tries,
        }
        if self.depth_range is not None:
            d["depth_range"] = list(self.depth_range)
        if self.len_range is not None:
            d["len_range"] = list(self.len_range)
        return d


@dataclass(frozen=True)
class OverlapConfig:
    """Controls the greedy support-set construction for arbitrary overlap."""

    support_size: tuple[int, int] = (2, 4)
    shared_codewords: int = 12
    max_restarts: int = 100
    candidate_trials: int = 10_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "support_size": list(self.support_size),
            "shared_codewords": self.shared_codewords,
            "max_restarts": self.max_restarts,
            "candidate_trials": self.candidate_trials,
        }


@dataclass(frozen=True)
class LanguageConfig:
    graph: str = "grid-4x4"
    code: CodeConfig = field(default_factory=CodeConfig)
    k: int | tuple[int, int] = 4
    assignment: str = "disjoint-random"
    overlap: OverlapConfig | None = None
    codeword_pool_file: str | None = None

    @property
    def k_range(self) -> tuple[int, int]:
        """Inclusive number of codewords assigned to each vertex."""
        return (self.k, self.k) if isinstance(self.k, int) else self.k

    @property
    def grid_n(self) -> int:
        """Side length of the grid graph described by :attr:`graph`."""
        m = _GRAPH_RE.match(self.graph.strip())
        if m is None:
            raise ConfigError(f"unsupported graph spec: {self.graph!r} (expected 'grid-NxN')")
        a, b = int(m.group(1)), int(m.group(2))
        if a != b:
            raise ConfigError(f"only square grids are supported, got {a}x{b}")
        return a

    @property
    def num_vertices(self) -> int:
        return self.grid_n**2

    def to_dict(self) -> dict[str, Any]:
        out = {
            "graph": self.graph,
            "code": self.code.to_dict(),
            "k": self.k if isinstance(self.k, int) else list(self.k),
            "assignment": self.assignment,
        }
        if self.codeword_pool_file is not None:
            out["codeword_pool_file"] = self.codeword_pool_file
        if self.overlap is not None:
            out["overlap"] = self.overlap.to_dict()
        return out


@dataclass(frozen=True)
class NoiseConfig:
    type: str
    gamma: float  # fraction of corrupted samples
    rho: float  # per-unit corruption rate inside a corrupted sample

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "gamma": self.gamma, "rho": self.rho}


@dataclass(frozen=True)
class DataConfig:
    walk_len: tuple[int, int] = (8, 32)
    pool_tokens: int = 10_000_000
    split: tuple[float, float, float] = (98.0, 1.0, 1.0)
    seed: int = 42
    reverse_walks: bool = False
    noise: NoiseConfig | None = None
    context_len: int = 512  # training context; used for packing + length warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "walk_len": list(self.walk_len),
            "pool_tokens": self.pool_tokens,
            "split": list(self.split),
            "seed": self.seed,
            "reverse_walks": self.reverse_walks,
            "noise": self.noise.to_dict() if self.noise else None,
            "context_len": self.context_len,
        }


@dataclass(frozen=True)
class TaskConfig:
    """Dataset objective; path-QA fields are ignored for language modeling."""

    type: str = "language-modeling"
    query_len: tuple[int, int] = (2, 5)
    segment_len: tuple[int, int] = (2, 16)
    path_trials: int = 1_000
    split_mode: str = "iid"
    pair_direction: str = "unordered"
    held_out_pairs: tuple[tuple[int, int], ...] = ()
    region_overlap_rows: int = 2

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.type == "path-qa":
            out.update(
                query_len=list(self.query_len),
                segment_len=list(self.segment_len),
                path_trials=self.path_trials,
                split_mode=self.split_mode,
                pair_direction=self.pair_direction,
                held_out_pairs=[list(pair) for pair in self.held_out_pairs],
                region_overlap_rows=self.region_overlap_rows,
            )
        return out


@dataclass(frozen=True)
class Config:
    language: LanguageConfig = field(default_factory=LanguageConfig)
    data: DataConfig = field(default_factory=DataConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language.to_dict(),
            "data": self.data.to_dict(),
            "task": self.task.to_dict(),
        }


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def _pair(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{name} must be a [min, max] pair, got {value!r}")
    lo, hi = int(value[0]), int(value[1])
    if lo > hi:
        raise ConfigError(f"{name}: min ({lo}) must be <= max ({hi})")
    return lo, hi


def _vertex_pair(value: Any, name: str) -> tuple[int, int]:
    """Parse an ordered vertex pair without treating it as a numeric range."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{name} must be a [u, v] pair, got {value!r}")
    return int(value[0]), int(value[1])


def _code_config(raw: dict[str, Any]) -> CodeConfig:
    unknown = set(raw) - {"type", "base_size", "depth_range", "len_range", "power_x", "max_tries"}
    if unknown:
        raise ConfigError(f"unknown code fields: {sorted(unknown)}")
    ctype = str(raw.get("type", "prefix-free"))
    if ctype not in CODE_TYPES:
        raise ConfigError(f"code.type must be one of {CODE_TYPES}, got {ctype!r}")
    return CodeConfig(
        type=ctype,
        base_size=int(raw.get("base_size", 3)),
        depth_range=_pair(raw["depth_range"], "code.depth_range") if "depth_range" in raw else None,
        len_range=_pair(raw["len_range"], "code.len_range") if "len_range" in raw else None,
        power_x=int(raw.get("power_x", 6)),
        max_tries=int(raw.get("max_tries", 10_000)),
    )


def _k_spec(value: Any) -> int | tuple[int, int]:
    """Parse fixed ``k`` or an inclusive per-vertex ``[min, max]`` range."""
    if isinstance(value, bool):
        raise ConfigError("language.k must be an integer or [min, max] pair")
    if isinstance(value, (list, tuple)):
        return _pair(value, "language.k")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ConfigError("language.k must be an integer or [min, max] pair") from None


def _overlap_config(raw: Any) -> OverlapConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("language.overlap must be a mapping")
    unknown = set(raw) - {
        "support_size", "shared_codewords", "max_restarts", "candidate_trials"
    }
    if unknown:
        raise ConfigError(f"unknown language.overlap fields: {sorted(unknown)}")
    return OverlapConfig(
        support_size=_pair(raw.get("support_size", [2, 4]), "language.overlap.support_size"),
        shared_codewords=int(raw.get("shared_codewords", 12)),
        max_restarts=int(raw.get("max_restarts", 100)),
        candidate_trials=int(raw.get("candidate_trials", 10_000)),
    )


def _task_config(raw: Any) -> TaskConfig:
    if raw is None:
        return TaskConfig()
    if not isinstance(raw, dict):
        raise ConfigError("task must be a mapping")
    unknown = set(raw) - {
        "type", "query_len", "segment_len", "path_trials", "split_mode",
        "pair_direction", "held_out_pairs", "region_overlap_rows",
    }
    if unknown:
        raise ConfigError(f"unknown task fields: {sorted(unknown)}")
    task_type = str(raw.get("type", "language-modeling"))
    if task_type not in TASK_TYPES:
        raise ConfigError(f"task.type must be one of {TASK_TYPES}, got {task_type!r}")
    if task_type == "language-modeling" and set(raw) - {"type"}:
        raise ConfigError("path-QA fields require task.type: path-qa")
    pairs_raw = raw.get("held_out_pairs", [])
    if not isinstance(pairs_raw, (list, tuple)):
        raise ConfigError("task.held_out_pairs must be a list of [u, v] pairs")
    pairs = tuple(_vertex_pair(pair, "task.held_out_pairs entry") for pair in pairs_raw)
    return TaskConfig(
        type=task_type,
        query_len=_pair(raw.get("query_len", [2, 5]), "task.query_len"),
        segment_len=_pair(raw.get("segment_len", [2, 16]), "task.segment_len"),
        path_trials=int(raw.get("path_trials", 1_000)),
        split_mode=str(raw.get("split_mode", "iid")),
        pair_direction=str(raw.get("pair_direction", "unordered")),
        held_out_pairs=pairs,
        region_overlap_rows=int(raw.get("region_overlap_rows", 2)),
    )


def _noise_config(raw: Any) -> NoiseConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("data.noise must be null or a mapping {type, gamma, rho}")
    ntype = str(raw.get("type", ""))
    if ntype not in NOISE_TYPES:
        raise ConfigError(f"data.noise.type must be one of {NOISE_TYPES}, got {ntype!r}")
    gamma, rho = float(raw.get("gamma", 0.0)), float(raw.get("rho", 0.0))
    for name, val in (("gamma", gamma), ("rho", rho)):
        if not 0.0 <= val <= 1.0:
            raise ConfigError(f"data.noise.{name} must be in [0, 1], got {val}")
    return NoiseConfig(type=ntype, gamma=gamma, rho=rho)


def parse_config(raw: dict[str, Any]) -> Config:
    """Build a validated :class:`Config` from a plain dict."""
    unknown = set(raw) - {"language", "data", "task"}
    if unknown:
        raise ConfigError(f"unknown top-level fields: {sorted(unknown)}")
    lang_raw = dict(raw.get("language") or {})
    data_raw = dict(raw.get("data") or {})

    unknown = set(lang_raw) - {
        "graph", "code", "k", "assignment", "overlap", "codeword_pool_file"
    }
    if unknown:
        raise ConfigError(f"unknown language fields: {sorted(unknown)}")
    unknown = set(data_raw) - {
        "walk_len", "pool_tokens", "split", "seed", "reverse_walks", "noise", "context_len"
    }
    if unknown:
        raise ConfigError(f"unknown data fields: {sorted(unknown)}")

    pool_raw = lang_raw.get("codeword_pool_file")
    codeword_pool_file = None if pool_raw is None else str(pool_raw).strip()
    assignment = str(lang_raw.get("assignment", "disjoint-random"))
    if assignment not in ASSIGNMENTS:
        raise ConfigError(f"language.assignment must be one of {ASSIGNMENTS}, got {assignment!r}")

    language = LanguageConfig(
        graph=str(lang_raw.get("graph", "grid-4x4")),
        code=_code_config(dict(lang_raw.get("code") or {})),
        k=_k_spec(lang_raw.get("k", 4)),
        assignment=assignment,
        overlap=_overlap_config(lang_raw.get("overlap")),
        codeword_pool_file=codeword_pool_file,
    )

    split = data_raw.get("split", [98, 1, 1])
    if not isinstance(split, (list, tuple)) or len(split) != 3:
        raise ConfigError("data.split must be a 3-element list [train, valid, test]")
    split_t = tuple(float(s) for s in split)

    reverse_walks = data_raw.get("reverse_walks", False)
    if not isinstance(reverse_walks, bool):
        raise ConfigError("data.reverse_walks must be true or false")

    data = DataConfig(
        walk_len=_pair(data_raw.get("walk_len", [8, 32]), "data.walk_len"),
        pool_tokens=int(data_raw.get("pool_tokens", 10_000_000)),
        split=split_t,  # type: ignore[arg-type]
        seed=int(data_raw.get("seed", 42)),
        reverse_walks=reverse_walks,
        noise=_noise_config(data_raw.get("noise")),
        context_len=int(data_raw.get("context_len", 512)),
    )

    cfg = Config(language=language, data=data, task=_task_config(raw.get("task")))
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> list[str]:
    """Hard-check structural constraints; return a list of soft warnings.

    Hard errors (raise :class:`ConfigError`):
      * malformed graph spec / non-square grid
      * ``base_size < 2``, ``power_x < 1``, or either ``k`` bound below 1
      * pool size sufficient for the maximum possible per-vertex assignment
      * degenerate walk length (``min < 2``) or split

    Soft warnings (returned, printed by the CLI):
      * worst-case sentence length exceeding ``data.context_len``
      * pool slack below 4x the required number of codewords
    """
    k_lo, k_hi = cfg.language.k_range
    if k_lo < 1:
        raise ConfigError("language.k must be >= 1 (or [min, max] with min >= 1)")
    n_v = cfg.language.num_vertices  # also validates the graph spec
    overlap = cfg.language.overlap
    if cfg.language.assignment == "arbitrary-overlap":
        if overlap is None:
            raise ConfigError(
                "language.overlap is required when assignment is 'arbitrary-overlap'"
            )
        support_lo, support_hi = overlap.support_size
        if support_lo < 2:
            raise ConfigError("language.overlap.support_size min must be >= 2")
        if support_hi > n_v:
            raise ConfigError(
                f"language.overlap.support_size max {support_hi} exceeds |V|={n_v}"
            )
        if overlap.shared_codewords < 1:
            raise ConfigError("language.overlap.shared_codewords must be >= 1")
        if overlap.max_restarts < 1 or overlap.candidate_trials < 1:
            raise ConfigError(
                "language.overlap.max_restarts and candidate_trials must be >= 1"
            )
        if k_hi < 2:
            raise ConfigError(
                "arbitrary-overlap needs k_max >= 2 so vertices have room for shared words"
            )
    elif overlap is not None:
        raise ConfigError(
            "language.overlap is only valid when assignment is 'arbitrary-overlap'"
        )

    external_pool = cfg.language.codeword_pool_file is not None
    if external_pool:
        if not cfg.language.codeword_pool_file:
            raise ConfigError("language.codeword_pool_file must not be empty")
    else:
        code = cfg.language.code
        lo, hi = code.length_range
        if lo < 1:
            raise ConfigError("code length/depth range must start at >= 1")
        if code.base_size < 2:
            raise ConfigError("code.base_size must be >= 2")
        if code.power_x < 1:
            raise ConfigError("code.power_x must be >= 1")
        pool_size = code.base_size**code.power_x
        needed = n_v * k_hi
        if pool_size < needed:
            raise ConfigError(
                f"codeword pool too small: |C|^x = {code.base_size}^{code.power_x} = {pool_size} "
                f"< |V| * k_max = {n_v} * {k_hi} = {needed}"
            )

    wlo, whi = cfg.data.walk_len
    if wlo < 2:
        raise ConfigError("data.walk_len min must be >= 2 (a walk has at least 2 vertices)")
    if cfg.data.pool_tokens < 1:
        raise ConfigError("data.pool_tokens must be >= 1")
    if sum(cfg.data.split) <= 0 or any(s < 0 for s in cfg.data.split):
        raise ConfigError("data.split must be non-negative with a positive sum")
    if cfg.data.context_len < 8:
        raise ConfigError("data.context_len must be >= 8")

    task = cfg.task
    if task.type == "path-qa":
        qlo, qhi = task.query_len
        slo, shi = task.segment_len
        if qlo < 2 or qhi > n_v:
            raise ConfigError(f"task.query_len must lie in [2, |V|={n_v}]")
        if slo < 2 or shi > n_v:
            raise ConfigError(f"task.segment_len must lie in [2, |V|={n_v}]")
        if task.path_trials < 1:
            raise ConfigError("task.path_trials must be >= 1")
        if task.split_mode not in QA_SPLIT_MODES:
            raise ConfigError(
                f"task.split_mode must be one of {QA_SPLIT_MODES}, got {task.split_mode!r}"
            )
        if task.pair_direction not in PAIR_DIRECTIONS:
            raise ConfigError(
                f"task.pair_direction must be one of {PAIR_DIRECTIONS}, "
                f"got {task.pair_direction!r}"
            )
        if task.split_mode == "held-out-pairs" and not task.held_out_pairs:
            raise ConfigError("held-out-pairs mode requires task.held_out_pairs")
        if task.split_mode != "held-out-pairs" and task.held_out_pairs:
            raise ConfigError(
                "task.held_out_pairs is only valid with split_mode: held-out-pairs"
            )
        for u, v in task.held_out_pairs:
            if u == v or not (0 <= u < n_v and 0 <= v < n_v):
                raise ConfigError(
                    f"invalid held-out pair {(u, v)} for vertices 0..{n_v - 1}"
                )
        pair_keys = [
            tuple(sorted(pair)) if task.pair_direction == "unordered" else pair
            for pair in task.held_out_pairs
        ]
        if len(set(pair_keys)) != len(pair_keys):
            raise ConfigError("task.held_out_pairs contains duplicate effective pairs")
        if task.split_mode == "cross-region":
            overlap_rows = task.region_overlap_rows
            grid_n = cfg.language.grid_n
            if not (1 <= overlap_rows < grid_n):
                raise ConfigError(
                    f"task.region_overlap_rows must lie in [1, {grid_n - 1}]"
                )
            if (grid_n - overlap_rows) % 2:
                raise ConfigError(
                    "grid rows minus region_overlap_rows must be even for a "
                    "symmetric cross-region split"
                )
            if (grid_n - overlap_rows) // 2 < 1:
                raise ConfigError("cross-region mode needs non-empty outer row bands")
        if cfg.data.noise is not None:
            raise ConfigError("data.noise is not yet supported for task.type: path-qa")
        if cfg.data.reverse_walks:
            raise ConfigError("data.reverse_walks is not applicable to task.type: path-qa")

    if external_pool:
        return []  # file-dependent checks and length warnings run during the build

    warnings: list[str] = []
    if pool_size < 4 * needed:
        warnings.append(
            f"pool slack is small: |C|^x = {pool_size} vs required {needed} "
            "(little room for future overlap constructions)"
        )
    max_word_len = hi * code.power_x
    if task.type == "path-qa":
        max_waypoints = task.query_len[1]
        max_answer_vertices = 1 + (max_waypoints - 1) * (task.segment_len[1] - 1)
        worst_bits = (
            max_waypoints * max_word_len + 1
            + max_answer_vertices * max_word_len + 2
        )
    else:
        worst_bits = whi * max_word_len + 2
    worst_description = (
        "max query plus piecewise-simple answer"
        if task.type == "path-qa"
        else f"walk_len max {whi} x codeword max {max_word_len} bits"
    )
    if worst_bits > cfg.data.context_len:
        warnings.append(
            f"worst-case sentence length {worst_bits} tokens ({worst_description}) "
            f"exceeds context_len {cfg.data.context_len}; "
            "actual codeword lengths depend on the sampled base code — check the build report"
        )
    return warnings


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return parse_config(raw)


def config_hash(cfg: Config) -> str:
    """Stable content hash of a config (sha256 of canonical JSON, 16 hex chars)."""
    canonical = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
