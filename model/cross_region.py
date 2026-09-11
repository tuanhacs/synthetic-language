"""Cross-region path-QA analysis at semantic and token-confidence levels."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

import _paths  # noqa: F401
from synthdata.qa import (
    QASample,
    cross_region_direction,
    cross_region_layout,
    cross_region_segment_count,
    validate_qa_answer,
)

from sample import generate


EVENTS = (
    "decodable",
    "valid_walk",
    "correct_start",
    "entered_overlap",
    "reached_target_region",
    "correct_first_overlap_exit",
    "returned_to_source_after_overlap",
    "hit_destination",
    "end_at_destination",
    "waypoints_in_order",
    "piecewise_simple",
    "semantic_success",
    "terminated",
)


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counter.values() if n)


def _relative_bands(sample: QASample, grid_n: int, overlap_rows: int):
    layout = cross_region_layout(grid_n, overlap_rows)
    direction = cross_region_direction(sample.query_vertices, grid_n, overlap_rows)
    if direction == "top-to-bottom":
        return direction, layout.top_only, layout.bottom_only, layout.upper & layout.lower
    if direction == "bottom-to-top":
        return direction, layout.bottom_only, layout.top_only, layout.upper & layout.lower
    raise ValueError(f"not a cross-region test query: {sample.query_vertices}")


def _walk_events(
    language,
    sample: QASample,
    answer_bits: str,
    terminated: bool,
    overlap_rows: int,
) -> dict[str, Any]:
    semantic = validate_qa_answer(language, answer_bits, sample.query_vertices)
    direction, source, target, overlap = _relative_bands(
        sample, language.graph.n, overlap_rows
    )
    walk = tuple(semantic["walk"])

    labels: list[str] = []
    for vertex in walk:
        if vertex in source:
            labels.append("source")
        elif vertex in target:
            labels.append("target")
        elif vertex in overlap:
            labels.append("overlap")
        else:
            labels.append("other")

    # Keep the transfer funnel nested. A decoded walk that starts at an
    # unrelated vertex in the target band must not count as evidence that the
    # model used the query destination to cross regions.
    correct_start = bool(semantic["correct_start"])
    entered = correct_start and "overlap" in labels[1:]
    first_overlap = labels.index("overlap", 1) if entered else None
    reached = bool(
        first_overlap is not None and "target" in labels[first_overlap + 1 :]
    )
    first_exit = None
    returned = False
    if first_overlap is not None:
        returned = "source" in labels[first_overlap + 1 :]
        for label in labels[first_overlap + 1 :]:
            if label != "overlap":
                first_exit = label
                break

    endpoint_region = labels[-1] if labels else "invalid"
    endpoint = walk[-1] if walk else None
    hit_destination = bool(
        reached and sample.query_vertices[-1] in walk[first_overlap + 1 :]
    )
    region_switches = sum(a != b for a, b in zip(labels, labels[1:]))
    steps_to_overlap = labels.index("overlap") if entered else None
    steps_to_target = labels.index("target") if reached else None

    return {
        "direction": direction,
        "decodable": bool(semantic["decodable"]),
        "valid_walk": bool(semantic["valid_walk"]),
        "correct_start": correct_start,
        "entered_overlap": entered,
        "reached_target_region": reached,
        "correct_first_overlap_exit": first_exit == "target",
        "returned_to_source_after_overlap": returned,
        "hit_destination": hit_destination,
        "end_at_destination": bool(
            hit_destination and endpoint == sample.query_vertices[-1]
        ),
        "waypoints_in_order": bool(semantic["waypoints_in_order"]),
        "piecewise_simple": bool(semantic["piecewise_simple"]),
        "semantic_success": bool(semantic["semantic_success"]),
        "terminated": terminated,
        "endpoint": endpoint,
        "endpoint_region": endpoint_region,
        "region_switches": region_switches,
        "steps_to_overlap": steps_to_overlap,
        "steps_to_target": steps_to_target,
    }


def _new_accumulator() -> dict[str, Any]:
    return {
        "n": 0,
        "events": Counter(),
        "endpoint_vertices": Counter(),
        "endpoint_regions": Counter(),
        "region_switches": 0,
        "steps_to_overlap": [],
        "steps_to_target": [],
    }


def _add(acc: dict[str, Any], event: dict[str, Any]) -> None:
    acc["n"] += 1
    for key in EVENTS:
        acc["events"][key] += int(event[key])
    acc["endpoint_vertices"][str(event["endpoint"])] += 1
    acc["endpoint_regions"][event["endpoint_region"]] += 1
    acc["region_switches"] += event["region_switches"]
    if event["steps_to_overlap"] is not None:
        acc["steps_to_overlap"].append(event["steps_to_overlap"])
    if event["steps_to_target"] is not None:
        acc["steps_to_target"].append(event["steps_to_target"])


def _mean(values: Sequence[int | float]) -> float | None:
    return sum(values) / len(values) if values else None


def _finalise(acc: dict[str, Any]) -> dict[str, Any]:
    n = acc["n"]
    if not n:
        return {"n": 0}
    out = {"n": n}
    out.update({f"{key}_pct": 100.0 * acc["events"][key] / n for key in EVENTS})
    started = acc["events"]["correct_start"]
    entered = acc["events"]["entered_overlap"]
    reached = acc["events"]["reached_target_region"]
    out["enter_overlap_given_correct_start_pct"] = (
        100.0 * entered / started if started else None
    )
    out["reach_target_given_overlap_pct"] = (
        100.0 * reached / entered if entered else None
    )
    out["correct_exit_given_overlap_pct"] = (
        100.0 * acc["events"]["correct_first_overlap_exit"] / entered
        if entered else None
    )
    out["return_to_source_given_overlap_pct"] = (
        100.0 * acc["events"]["returned_to_source_after_overlap"] / entered
        if entered else None
    )
    out["hit_destination_given_target_region_pct"] = (
        100.0 * acc["events"]["hit_destination"] / reached if reached else None
    )
    hit = acc["events"]["hit_destination"]
    out["end_at_destination_given_hit_pct"] = (
        100.0 * acc["events"]["end_at_destination"] / hit if hit else None
    )
    out["semantic_success_given_correct_start_pct"] = (
        100.0 * acc["events"]["semantic_success"] / started
        if started else None
    )
    out["mean_region_switches"] = acc["region_switches"] / n
    out["mean_steps_to_overlap_when_reached"] = _mean(acc["steps_to_overlap"])
    out["mean_steps_to_target_region_when_reached"] = _mean(acc["steps_to_target"])
    out["endpoint_region_distribution"] = {
        key: count / n for key, count in sorted(acc["endpoint_regions"].items())
    }
    out["endpoint_region_entropy_bits"] = _entropy(acc["endpoint_regions"])
    out["endpoint_vertex_entropy_bits"] = _entropy(acc["endpoint_vertices"])
    out["most_common_endpoints"] = acc["endpoint_vertices"].most_common(10)
    return out


def _group_key(sample: QASample, grid_n: int, overlap_rows: int) -> dict[str, str]:
    start, end = sample.query_vertices[0], sample.query_vertices[-1]
    sr, sc = divmod(start, grid_n)
    er, ec = divmod(end, grid_n)
    return {
        "direction": str(cross_region_direction(sample.query_vertices, grid_n, overlap_rows)),
        "cross_segments": str(
            cross_region_segment_count(sample.query_vertices, grid_n, overlap_rows)
        ),
        "query_length": str(len(sample.query_vertices)),
        "manhattan_distance": str(abs(sr - er) + abs(sc - ec)),
    }


def _private_words(language) -> dict[int, tuple[str, ...]]:
    owners: dict[str, set[int]] = defaultdict(set)
    for vertex, book in enumerate(language.codebooks.books):
        for word in book:
            owners[word].add(vertex)
    return {
        vertex: tuple(word for word in book if owners[word] == {vertex})
        for vertex, book in enumerate(language.codebooks.books)
    }


def _token_confidence(
    model,
    tokenizer,
    samples: Sequence[QASample],
    grid_n: int,
    overlap_rows: int,
    device,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Teacher-forced confidence on codewords at the two region gates."""
    collected: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"codewords": 0, "bits": 0, "chosen_prob": 0.0, "entropy": 0.0, "margin": 0.0}
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            ids = [tokenizer.encode(sample.bits) for sample in chunk]
            width = max(map(len, ids))
            batch = torch.full((len(ids), width), tokenizer.PAD, dtype=torch.long, device=device)
            for row, seq in enumerate(ids):
                batch[row, : len(seq)] = torch.tensor(seq, device=device)
            logits = model(batch).float()

            for row, (sample, seq) in enumerate(zip(chunk, ids)):
                _, source, target, overlap = _relative_bands(sample, grid_n, overlap_rows)
                labels = [
                    "source" if v in source else "target" if v in target else "overlap"
                    for v in sample.answer_walk
                ]
                gate_indices: dict[str, int] = {}
                for index, (left, right) in enumerate(zip(labels, labels[1:]), start=1):
                    if left == "source" and right == "overlap":
                        gate_indices.setdefault("enter_overlap", index)
                    if left == "overlap" and right == "target":
                        gate_indices.setdefault("exit_to_target", index)

                answer_offset = len(sample.query_bits) + 2  # BOS + query + '_'
                for gate, vertex_index in gate_indices.items():
                    bit_start = 0 if vertex_index == 0 else sample.answer_cuts[vertex_index - 1]
                    bit_end = sample.answer_cuts[vertex_index]
                    stats = collected[gate]
                    stats["codewords"] += 1
                    for bit_index in range(bit_start, bit_end):
                        token_position = answer_offset + bit_index
                        scores = logits[row, token_position - 1].clone()
                        scores[tokenizer.BOS] = float("-inf")
                        scores[tokenizer.PAD] = float("-inf")
                        scores[tokenizer.SEP] = float("-inf")
                        probs = torch.softmax(scores, dim=-1)
                        chosen = seq[token_position]
                        finite = probs[probs > 0]
                        top = torch.topk(probs, k=2).values
                        stats["bits"] += 1
                        stats["chosen_prob"] += float(probs[chosen])
                        stats["entropy"] += float(-(finite * finite.log2()).sum())
                        stats["margin"] += float(top[0] - top[1])

                eos_position = len(seq) - 1
                scores = logits[row, eos_position - 1].clone()
                scores[tokenizer.BOS] = float("-inf")
                scores[tokenizer.PAD] = float("-inf")
                scores[tokenizer.SEP] = float("-inf")
                probs = torch.softmax(scores, dim=-1)
                finite = probs[probs > 0]
                top = torch.topk(probs, k=2).values
                stats = collected["eos_at_reference_end"]
                stats["codewords"] += 1
                stats["bits"] += 1
                stats["chosen_prob"] += float(probs[tokenizer.EOS])
                stats["entropy"] += float(-(finite * finite.log2()).sum())
                stats["margin"] += float(top[0] - top[1])

    return {
        name: {
            "codewords": values["codewords"],
            "bits": values["bits"],
            "mean_chosen_token_probability": values["chosen_prob"] / values["bits"],
            "mean_entropy_bits": values["entropy"] / values["bits"],
            "mean_top1_top2_margin": values["margin"] / values["bits"],
        }
        for name, values in collected.items()
        if values["bits"]
    }


def analyze_cross_region(
    model,
    data,
    n_queries: int = 500,
    rollouts_per_query: int = 32,
    query_batch_size: int = 16,
    temperature: float = 1.0,
    max_len: int | None = None,
    gen_batch_size: int = 512,
    confidence_queries: int = 200,
    intervention_rollouts: int = 8,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Estimate region-transfer probabilities and uncertainty by repeated rollouts."""
    task = data.manifest.get("config", {}).get("task", {})
    if task.get("split_mode") != "cross-region":
        raise ValueError("cross-region analysis requires split_mode: cross-region")
    overlap_rows = int(task.get("region_overlap_rows", 2))
    grid_n = data.language.graph.n
    candidates = list(data.test_samples)
    random.Random(seed).shuffle(candidates)
    selected = candidates[: min(n_queries, len(candidates))]
    if not selected:
        raise ValueError("cross-region analysis needs test samples")

    rng = torch.Generator(device="cpu").manual_seed(seed)
    overall = _new_accumulator()
    grouped: dict[str, dict[str, dict[str, Any]]] = {
        name: defaultdict(_new_accumulator)
        for name in ("direction", "cross_segments", "query_length", "manhattan_distance")
    }
    query_records: list[dict[str, Any]] = []
    max_len = max_len or model.cfg.context_len

    # Equal-length prompts share one KV-cache batch. Bucketing here avoids the
    # generator having to split a mixed batch into many tiny serial calls.
    prompt_buckets: dict[int, list[QASample]] = defaultdict(list)
    for sample in selected:
        prompt_buckets[len(sample.query_bits)].append(sample)
    chunks = [
        bucket[start : start + query_batch_size]
        for bucket in prompt_buckets.values()
        for start in range(0, len(bucket), query_batch_size)
    ]

    completed_queries = 0
    for chunk_index, chunk in enumerate(chunks, start=1):
        prefixes = [sample.query_bits + "_" for sample in chunk for _ in range(rollouts_per_query)]
        generated = generate(
            model,
            data.tokenizer,
            n=len(prefixes),
            temperature=temperature,
            max_len=max_len,
            prefix_bits=prefixes,
            device=device,
            rng=rng,
            batch_size=gen_batch_size,
            forbid_separator=True,
        )
        for local_index, sample in enumerate(chunk):
            begin = local_index * rollouts_per_query
            completions = generated[begin : begin + rollouts_per_query]
            prompt = sample.query_bits + "_"
            query_acc = _new_accumulator()
            for full in completions:
                answer = full[len(prompt) :]
                terminated = len(full) < max_len - 1
                event = _walk_events(
                    data.language, sample, answer, terminated, overlap_rows
                )
                _add(overall, event)
                _add(query_acc, event)
                for group_name, group_value in _group_key(
                    sample, grid_n, overlap_rows
                ).items():
                    _add(grouped[group_name][group_value], event)

            query_summary = _finalise(query_acc)
            query_records.append(
                {
                    "query_vertices": list(sample.query_vertices),
                    "query_bits": sample.query_bits,
                    **_group_key(sample, grid_n, overlap_rows),
                    "rollouts": query_summary,
                }
            )
        completed_queries += len(chunk)
        if chunk_index == 1 or chunk_index % 10 == 0 or chunk_index == len(chunks):
            print(
                f"cross-region rollouts: {completed_queries}/{len(selected)} queries",
                flush=True,
            )

    per_query_cross = [r["rollouts"]["reached_target_region_pct"] / 100 for r in query_records]
    per_query_success = [r["rollouts"]["semantic_success_pct"] / 100 for r in query_records]
    per_query_endpoint_entropy = [
        r["rollouts"]["endpoint_vertex_entropy_bits"] for r in query_records
    ]

    # Destination intervention on two-waypoint queries: keep the source surface
    # codeword, replace only the destination with a private codeword in the
    # source band, and measure how much target-region reach probability drops.
    intervention: dict[str, Any] = {"n_queries": 0}
    pair_samples = [sample for sample in selected if len(sample.query_vertices) == 2]
    if intervention_rollouts > 0 and pair_samples:
        private = _private_words(data.language)
        control_prefixes: list[str] = []
        original_probs: list[float] = []
        control_meta: list[tuple[QASample, str]] = []
        py_rng = random.Random(seed + 1)
        for sample in pair_samples:
            _, source_band, _, _ = _relative_bands(sample, grid_n, overlap_rows)
            choices = sorted(
                vertex
                for vertex in source_band
                if vertex != sample.query_vertices[0] and private.get(vertex)
            )
            if not choices:
                continue
            control_destination = py_rng.choice(choices)
            source_surface = sample.query_bits[: sample.query_cuts[0]]
            control_query = source_surface + private[control_destination][0]
            control_prefixes.extend([control_query + "_"] * intervention_rollouts)
            original_record = next(
                record for record in query_records
                if record["query_bits"] == sample.query_bits
            )
            original_probs.append(
                original_record["rollouts"]["reached_target_region_pct"] / 100
            )
            control_meta.append((sample, control_query + "_"))

        controls = generate(
            model,
            data.tokenizer,
            n=len(control_prefixes),
            temperature=temperature,
            max_len=max_len,
            prefix_bits=control_prefixes,
            device=device,
            rng=rng,
            batch_size=gen_batch_size,
            forbid_separator=True,
        )
        control_probs: list[float] = []
        for index, (original, prompt) in enumerate(control_meta):
            _, _, original_target, overlap = _relative_bands(
                original, grid_n, overlap_rows
            )
            reached = 0
            for full in controls[
                index * intervention_rollouts : (index + 1) * intervention_rollouts
            ]:
                answer = full[len(prompt) :]
                decoded = data.language.decode(answer)
                walk = decoded[0] if decoded else []
                first_overlap = next(
                    (i for i, vertex in enumerate(walk[1:], start=1) if vertex in overlap),
                    None,
                )
                reached += bool(
                    walk
                    and walk[0] == original.query_vertices[0]
                    and first_overlap is not None
                    and any(v in original_target for v in walk[first_overlap + 1 :])
                )
            control_probs.append(reached / intervention_rollouts)
        intervention = {
            "n_queries": len(control_meta),
            "rollouts_per_control": intervention_rollouts,
            "original_opposite_destination_mean_reach_probability": _mean(original_probs),
            "same_region_destination_mean_reach_probability": _mean(control_probs),
            "destination_influence_score": _mean(
                [a - b for a, b in zip(original_probs, control_probs)]
            ),
        }

    confidence = _token_confidence(
        model,
        data.tokenizer,
        selected[:confidence_queries],
        grid_n,
        overlap_rows,
        device,
    ) if confidence_queries > 0 else {}

    report = {
        "config_hash": data.manifest.get("config_hash"),
        "n_queries": len(selected),
        "rollouts_per_query": rollouts_per_query,
        "total_rollouts": len(selected) * rollouts_per_query,
        "temperature": temperature,
        "seed": seed,
        "region_overlap_rows": overlap_rows,
        "overall": _finalise(overall),
        "per_query_uncertainty": {
            "mean_cross_probability": _mean(per_query_cross),
            "mean_semantic_success_probability": _mean(per_query_success),
            "mean_endpoint_vertex_entropy_bits": _mean(per_query_endpoint_entropy),
        },
        "by": {
            name: {key: _finalise(acc) for key, acc in sorted(groups.items())}
            for name, groups in grouped.items()
        },
        "destination_intervention": intervention,
        "teacher_forced_token_confidence": confidence,
    }
    return report, query_records


def save_cross_region_analysis(
    out_dir: str | Path, report: dict[str, Any], queries: Sequence[dict[str, Any]]
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "cross_region_analysis.json"
    queries_path = out / "cross_region_queries.jsonl"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with queries_path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query, separators=(",", ":")) + "\n")
    return report_path, queries_path
