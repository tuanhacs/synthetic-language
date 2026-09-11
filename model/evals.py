"""Evaluations (context.md §8, v1 scope).

Three instruments, all scored against exact ground truth:

* **validity** — free generation (``cut = 0``) and prefix completion (``cut = k``),
  multinomial sampling at temperature ``tau``, scored with ``Language.is_valid``
  (the exact DP decoder, the same instrument for every regime);
* **loss vs entropy floor** — valid/test cross-entropy in bits/token and the gap to
  the Bayes-optimal floor recorded in the dataset manifest;
* **diversity / memorisation** — distinct-sentence fraction and the fraction of
  generated sentences that appear verbatim in the training split.

Losses are computed **sentence-aligned** (``BOS + bits + EOS``, PAD-filled, BOS/PAD
targets masked out) so the token accounting is exactly the one the entropy floor is
defined over: every ``bits`` token plus the final EOS, no BOS, no cross-sentence
truncation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

import _paths  # noqa: F401  (sys.path shim for synthdata; must precede synthdata imports)
from synthdata.language import Language, Sample
from synthdata.qa import (
    QASample,
    cross_region_direction,
    cross_region_segment_count,
    validate_qa_answer,
)

from config import EvalConfig
from data import IGNORE_INDEX, PackedData, sentence_batches
from model import Model
from sample import generate

LN2 = math.log(2.0)


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #


@torch.no_grad()
def masked_ce(
    model: Model,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str = "cpu",
) -> tuple[float, int]:
    """``(total nats, number of scored tokens)`` over the given batches."""
    model.eval()
    total, count = 0.0, 0
    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        total += float(loss)
        count += int((targets != IGNORE_INDEX).sum())
    return total, count


def bits_per_token(
    model: Model,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str = "cpu",
) -> float:
    total, count = masked_ce(model, batches, device)
    if count == 0:
        return float("nan")
    return total / count / LN2


def eval_loss(
    model: Model,
    data: PackedData,
    splits: Sequence[str] = ("valid", "test"),
    batch_size: int = 32,
    device: torch.device | str = "cpu",
) -> dict:
    """Sentence-aligned loss in bits/token per split, plus the gap to the floor."""
    floor = data.entropy_floor
    out: dict = {"entropy_floor_bits_per_token": floor, "splits": {}}
    for name in splits:
        samples = data.samples(name)
        if not samples:
            continue
        loss = bits_per_token(
            model, sentence_batches(samples, batch_size, data.tokenizer), device
        )
        out["splits"][name] = {
            "num_sentences": len(samples),
            "bits_per_token": loss,
            "gap_to_floor": None if floor is None else loss - floor,
        }
    return out


# --------------------------------------------------------------------------- #
# validity
# --------------------------------------------------------------------------- #


def eval_validity(
    model: Model,
    language: Language,
    tokenizer,
    n_samples: int = 200,
    temperature: float = 1.0,
    cut: int = 0,
    test_samples: Sequence[Sample] = (),
    max_len: int | None = None,
    device: torch.device | str = "cpu",
    rng: torch.Generator | None = None,
    batch_size: int = 64,
) -> dict:
    """Generate (``cut=0``) or complete (``cut=k``) and score with the exact decoder.

    For completions the prefixes are the first ``cut`` bits of fresh *test* sentences
    (never seen in training). Sentences shorter than ``cut + 1`` bits are skipped.

    Reported:
      * ``decodable_pct`` — the bit string can be fully segmented into global
        codewords, without checking vertex ownership or graph adjacency;
      * ``validity_pct`` — ``language.is_valid(full_bits)``;
      * ``terminated_pct`` — the model emitted EOS itself instead of running into
        ``max_len``;
      * ``valid_and_terminated_pct`` — the model's *own* EOS placement yields a valid
        sentence (the strict reading of "did it finish a real sentence?");
      * ``walk_len_in_range_pct`` — among valid strings, those whose decoded walk
        length lies inside the language's ``walk_len`` range (``is_valid`` itself
        imposes no length constraint).
    """
    if cut == 0:
        prefixes = None
        n = n_samples
    else:
        pool = [s.bits for s in test_samples if len(s.bits) > cut]
        if not pool:
            raise ValueError(f"no test sentence longer than cut={cut} bits")
        prefixes = [pool[i % len(pool)][:cut] for i in range(n_samples)]
        n = n_samples

    bits = generate(
        model,
        tokenizer,
        n=n,
        temperature=temperature,
        max_len=max_len,
        prefix_bits=prefixes,
        device=device,
        rng=rng,
        batch_size=batch_size,
    )

    lo, hi = language.walk_len
    decodable, valid, terminated, valid_term, in_range = 0, 0, 0, 0, 0
    limit = (max_len or model.cfg.context_len) - 1 - (cut if cut else 0)
    for i, string in enumerate(bits):
        produced = len(string) - (cut if cut else 0)
        stopped_by_eos = produced < limit
        can_decode = language.is_decodable(string)
        ok = language.is_valid(string)
        decodable += can_decode
        valid += ok
        terminated += stopped_by_eos
        valid_term += ok and stopped_by_eos
        if ok:
            walks = language.decode(string)
            if walks and lo <= len(walks[0]) <= hi:
                in_range += 1

    total = len(bits)
    return {
        "cut": cut,
        "temperature": temperature,
        "n_samples": total,
        "decodable_pct": 100.0 * decodable / total,
        "validity_pct": 100.0 * valid / total,
        "terminated_pct": 100.0 * terminated / total,
        "valid_and_terminated_pct": 100.0 * valid_term / total,
        "walk_len_in_range_pct": 100.0 * in_range / total,
        "mean_generated_bits": sum(len(s) for s in bits) / total,
        "samples": bits,
    }


# --------------------------------------------------------------------------- #
# diversity / memorisation
# --------------------------------------------------------------------------- #


def eval_diversity(generated_bits: Sequence[str], train_bits: Iterable[str] = ()) -> dict:
    """Distinct-sentence fraction and verbatim overlap with the training split."""
    if not generated_bits:
        return {"n": 0}
    train_set = set(train_bits)
    distinct = len(set(generated_bits))
    memorised = sum(1 for s in generated_bits if s in train_set)
    counts: dict[str, int] = {}
    for s in generated_bits:
        counts[s] = counts.get(s, 0) + 1
    collisions = sum(c - 1 for c in counts.values())
    return {
        "n": len(generated_bits),
        "distinct": distinct,
        "distinct_frac": distinct / len(generated_bits),
        "collisions": collisions,
        "memorised": memorised,
        "memorisation_frac": memorised / len(generated_bits) if train_set else None,
    }


# --------------------------------------------------------------------------- #
# path QA
# --------------------------------------------------------------------------- #


def eval_path_qa(
    model: Model,
    language: Language,
    tokenizer,
    test_samples: Sequence[QASample],
    n_samples: int = 200,
    temperature: float = 1.0,
    max_len: int | None = None,
    device: torch.device | str = "cpu",
    rng: torch.Generator | None = None,
    batch_size: int = 64,
    task_config: dict | None = None,
) -> dict:
    """Prompt with ``query_bits + '_'`` and score any semantically valid route."""
    if not test_samples:
        raise ValueError("path-QA evaluation needs a non-empty test split")
    selected = [test_samples[i % len(test_samples)] for i in range(n_samples)]
    prefixes = [sample.query_bits + "_" for sample in selected]
    generated = generate(
        model,
        tokenizer,
        n=n_samples,
        temperature=temperature,
        max_len=max_len,
        prefix_bits=prefixes,
        device=device,
        rng=rng,
        batch_size=batch_size,
        forbid_separator=True,
    )
    limit = max_len or model.cfg.context_len
    counts = {
        "answer_decodable": 0,
        "valid_walk": 0,
        "correct_start": 0,
        "correct_end": 0,
        "waypoints_in_order": 0,
        "piecewise_simple": 0,
        "semantic_success": 0,
        "full_success": 0,
        "exact_reference_bits": 0,
        "exact_reference_walk": 0,
        "terminated": 0,
    }
    answers: list[str] = []
    examples: list[dict] = []
    answer_vertices = 0
    decoded_answers = 0
    cross_enabled = bool(
        task_config and task_config.get("split_mode") == "cross-region"
    )
    overlap_rows = int((task_config or {}).get("region_overlap_rows", 2))
    by_cross_count: dict[int, dict[str, int]] = {}
    by_direction: dict[str, dict[str, int]] = {}
    for sample, prefix, full in zip(selected, prefixes, generated):
        if not full.startswith(prefix):
            raise RuntimeError("generator did not preserve the QA prompt")
        answer = full[len(prefix) :]
        answers.append(answer)
        terminated = len(full) < limit - 1  # BOS is not present in returned text
        result = validate_qa_answer(language, answer, sample.query_vertices)
        counts["answer_decodable"] += result["decodable"]
        for key in (
            "valid_walk", "correct_start", "correct_end", "waypoints_in_order",
            "piecewise_simple", "semantic_success",
        ):
            counts[key] += result[key]
        counts["terminated"] += terminated
        counts["full_success"] += result["semantic_success"] and terminated
        counts["exact_reference_bits"] += answer == sample.answer_bits
        counts["exact_reference_walk"] += result["walk"] == sample.answer_walk
        cross_segments = 0
        direction = None
        if cross_enabled:
            cross_segments = cross_region_segment_count(
                sample.query_vertices, language.graph.n, overlap_rows
            )
            direction = cross_region_direction(
                sample.query_vertices, language.graph.n, overlap_rows
            )
            for groups, key in (
                (by_cross_count, cross_segments),
                (by_direction, direction or "other"),
            ):
                bucket = groups.setdefault(
                    key, {"n": 0, "valid_walk": 0, "semantic_success": 0, "full_success": 0}
                )
                bucket["n"] += 1
                bucket["valid_walk"] += int(result["valid_walk"])
                bucket["semantic_success"] += int(result["semantic_success"])
                bucket["full_success"] += int(result["semantic_success"] and terminated)
        if result["walk"]:
            decoded_answers += 1
            answer_vertices += len(result["walk"])
        if len(examples) < 10:
            examples.append(
                {
                    "query_vertices": list(sample.query_vertices),
                    "query_bits": sample.query_bits,
                    "reference_walk": list(sample.answer_walk),
                    "generated_answer_bits": answer,
                    "decoded_walk": list(result["walk"]),
                    "terminated": terminated,
                    "semantic_success": result["semantic_success"],
                    **(
                        {
                            "cross_segments": cross_segments,
                            "cross_direction": direction,
                        }
                        if cross_enabled
                        else {}
                    ),
                }
            )
    total = len(selected)
    report = {
        "temperature": temperature,
        "n_samples": total,
        **{f"{key}_pct": 100.0 * value / total for key, value in counts.items()},
        "mean_generated_answer_bits": sum(map(len, answers)) / total,
        "mean_decoded_answer_vertices": (
            answer_vertices / decoded_answers if decoded_answers else None
        ),
        "examples": examples,
    }
    if cross_enabled:
        def summarise(groups: dict) -> dict[str, dict[str, float | int]]:
            return {
                str(key): {
                    "n": bucket["n"],
                    "valid_walk_pct": 100.0 * bucket["valid_walk"] / bucket["n"],
                    "semantic_success_pct": (
                        100.0 * bucket["semantic_success"] / bucket["n"]
                    ),
                    "full_success_pct": 100.0 * bucket["full_success"] / bucket["n"],
                }
                for key, bucket in sorted(groups.items(), key=lambda item: str(item[0]))
            }

        report["cross_region"] = {
            "region_overlap_rows": overlap_rows,
            "mean_cross_segments": sum(
                count * bucket["n"] for count, bucket in by_cross_count.items()
            ) / total,
            "by_cross_segments": summarise(by_cross_count),
            "by_direction": summarise(by_direction),
        }
    return report


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def run_all(
    model: Model,
    data: PackedData,
    eval_cfg: EvalConfig,
    device: torch.device | str = "cpu",
    out_dir: str | Path | None = None,
    keep_samples: int = 10,
    extra: dict | None = None,
) -> dict:
    """Run every v1 evaluation and (optionally) write ``eval_report.json``."""
    rng = torch.Generator(device="cpu").manual_seed(eval_cfg.seed)
    report: dict = {
        "dataset_dir": str(data.dataset_dir),
        "config_hash": data.manifest.get("config_hash"),
        "regime": data.manifest.get("certification", {}).get("regime"),
        "task": data.manifest.get("config", {}).get(
            "task", {"type": "language-modeling"}
        ),
        "model": model.cfg.to_dict(),
        "num_params": model.num_params(),
        "eval": eval_cfg.to_dict(),
        "loss": eval_loss(model, data, device=device),
        "generation": {},
    }
    if extra:
        report.update(extra)

    task_type = data.manifest.get("config", {}).get("task", {}).get(
        "type", "language-modeling"
    )
    if task_type == "path-qa":
        report["generation"]["path_qa"] = eval_path_qa(
            model,
            data.language,
            data.tokenizer,
            data.test_samples,
            n_samples=eval_cfg.n_samples,
            temperature=eval_cfg.temperature,
            max_len=eval_cfg.max_len,
            device=device,
            rng=rng,
            batch_size=eval_cfg.gen_batch_size,
            task_config=report["task"],
        )
        if out_dir is not None:
            path = Path(out_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "eval_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            report["report_path"] = str(path / "eval_report.json")
        return report

    train_bits = [s.bits for s in data.train_samples]
    for cut in eval_cfg.cuts:
        res = eval_validity(
            model,
            data.language,
            data.tokenizer,
            n_samples=eval_cfg.n_samples,
            temperature=eval_cfg.temperature,
            cut=cut,
            test_samples=data.test_samples,
            max_len=eval_cfg.max_len,
            device=device,
            rng=rng,
            batch_size=eval_cfg.gen_batch_size,
        )
        samples = res.pop("samples")
        res["diversity"] = eval_diversity(samples, train_bits)
        res["examples"] = samples[:keep_samples]
        report["generation"][f"cut{cut}"] = res

    if out_dir is not None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "eval_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        report["report_path"] = str(path / "eval_report.json")
    return report


def format_report(report: dict) -> str:
    """Human-readable rendering of :func:`run_all`'s output."""
    lines = []
    lines.append(f"dataset      {report['dataset_dir']}  ({report.get('regime')})")
    lines.append(f"model        {report['model']['size'] or 'custom'}  {report['num_params']:,} params")
    floor = report["loss"]["entropy_floor_bits_per_token"]
    lines.append(f"entropy floor {floor:.4f} bits/token" if floor else "entropy floor unknown")
    for name, res in report["loss"]["splits"].items():
        gap = res["gap_to_floor"]
        gap_s = "n/a" if gap is None else f"{gap:+.4f}"
        lines.append(
            f"  {name:<5} loss {res['bits_per_token']:.4f} bits/token   gap {gap_s}"
            f"   ({res['num_sentences']} sentences)"
        )
    for key, res in report["generation"].items():
        if key == "path_qa":
            lines.append(
                f"  path-QA tau={res['temperature']} n={res['n_samples']}  "
                f"decodable {res['answer_decodable_pct']:.1f}%  "
                f"valid-walk {res['valid_walk_pct']:.1f}%  "
                f"semantic {res['semantic_success_pct']:.1f}%  "
                f"full+EOS {res['full_success_pct']:.1f}%"
            )
            lines.append(
                f"         start {res['correct_start_pct']:.1f}%  "
                f"end {res['correct_end_pct']:.1f}%  "
                f"waypoints {res['waypoints_in_order_pct']:.1f}%  "
                f"piecewise-simple {res['piecewise_simple_pct']:.1f}%  "
                f"terminated {res['terminated_pct']:.1f}%"
            )
            if "cross_region" in res:
                cross = res["cross_region"]
                lines.append(
                    f"         cross segments mean={cross['mean_cross_segments']:.2f}  "
                    f"by-count={cross['by_cross_segments']}"
                )
                lines.append(f"         by-direction={cross['by_direction']}")
            continue
        div = res["diversity"]
        mem = div["memorisation_frac"]
        decodable = res.get("decodable_pct")
        decodable_s = "n/a" if decodable is None else f"{decodable:.1f}%"
        lines.append(
            f"  {key:<6} tau={res['temperature']}  n={res['n_samples']}  "
            f"decodable {decodable_s}  valid {res['validity_pct']:.1f}%  "
            f"valid+EOS {res['valid_and_terminated_pct']:.1f}%  "
            f"terminated {res['terminated_pct']:.1f}%  walk-len-in-range {res['walk_len_in_range_pct']:.1f}%"
        )
        lines.append(
            f"         distinct {div['distinct_frac']:.3f}  "
            f"memorised {'n/a' if mem is None else f'{mem:.3f}'}  "
            f"mean bits {res['mean_generated_bits']:.1f}"
        )
    return "\n".join(lines)
