"""Path-QA samples and piecewise-simple route generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from .config import DataConfig, TaskConfig
from .dataset import Splits
from .language import Language


@dataclass(frozen=True)
class QASample:
    """Encoded waypoints, ``_``, and an encoded piecewise-simple answer walk."""

    query_bits: str
    answer_bits: str
    query_vertices: tuple[int, ...]
    answer_walk: tuple[int, ...]
    query_cuts: tuple[int, ...]
    answer_cuts: tuple[int, ...]
    segment_cuts: tuple[int, ...]

    @property
    def bits(self) -> str:
        return f"{self.query_bits}_{self.answer_bits}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_bits": self.query_bits,
            "answer_bits": self.answer_bits,
            "query_vertices": list(self.query_vertices),
            "answer_walk": list(self.answer_walk),
            "query_cuts": list(self.query_cuts),
            "answer_cuts": list(self.answer_cuts),
            "segment_cuts": list(self.segment_cuts),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "QASample":
        return QASample(
            query_bits=str(d["query_bits"]),
            answer_bits=str(d["answer_bits"]),
            query_vertices=tuple(int(v) for v in d["query_vertices"]),
            answer_walk=tuple(int(v) for v in d["answer_walk"]),
            query_cuts=tuple(int(v) for v in d["query_cuts"]),
            answer_cuts=tuple(int(v) for v in d["answer_cuts"]),
            segment_cuts=tuple(int(v) for v in d["segment_cuts"]),
        )


def _pair_key(pair: tuple[int, int], direction: str) -> tuple[int, int]:
    return tuple(sorted(pair)) if direction == "unordered" else pair


def query_contains_held_out(query: Sequence[int], task: TaskConfig) -> bool:
    held = {_pair_key(pair, task.pair_direction) for pair in task.held_out_pairs}
    return any(
        _pair_key((u, v), task.pair_direction) in held
        for u, v in zip(query, query[1:])
    )


def waypoints_in_order(walk: Sequence[int], query: Sequence[int]) -> bool:
    """Whether ``query`` is a subsequence of ``walk`` with matching endpoints."""
    if not walk or not query or walk[0] != query[0] or walk[-1] != query[-1]:
        return False
    cursor = 0
    for vertex in walk:
        if cursor < len(query) and vertex == query[cursor]:
            cursor += 1
    return cursor == len(query)


def is_piecewise_simple_answer(walk: Sequence[int], query: Sequence[int]) -> bool:
    """Can the walk be partitioned at query waypoints into simple segments?

    A vertex may repeat across different segments. If a waypoint occurs several
    times, dynamic programming accepts the answer when at least one ordered set
    of boundary occurrences makes every individual segment simple.
    """
    if len(query) < 2 or not walk or walk[0] != query[0] or walk[-1] != query[-1]:
        return False
    positions = {0}
    for waypoint in query[1:]:
        next_positions: set[int] = set()
        for start in positions:
            seen = {walk[start]}
            for end in range(start + 1, len(walk)):
                vertex = walk[end]
                if vertex in seen:
                    break
                seen.add(vertex)
                if vertex == waypoint:
                    next_positions.add(end)
        positions = next_positions
        if not positions:
            return False
    return len(walk) - 1 in positions


def validate_qa_answer(
    language: Language, answer_bits: str, query: Sequence[int]
) -> dict[str, Any]:
    """Semantic QA checks; any valid alternative route is accepted."""
    decodable = language.is_decodable(answer_bits)
    walks = language.decode(answer_bits) if decodable else []
    valid = bool(walks)
    walk = tuple(walks[0]) if valid else ()
    correct_start = valid and walk[0] == query[0]
    correct_end = valid and walk[-1] == query[-1]
    ordered = valid and waypoints_in_order(walk, query)
    piecewise_simple = valid and is_piecewise_simple_answer(walk, query)
    return {
        "decodable": decodable,
        "valid_walk": valid,
        "correct_start": correct_start,
        "correct_end": correct_end,
        "waypoints_in_order": ordered,
        "piecewise_simple": piecewise_simple,
        "semantic_success": bool(correct_start and correct_end and piecewise_simple),
        "walk": walk,
    }


def random_simple_path(
    language: Language,
    start: int,
    end: int,
    length_range: tuple[int, int],
    rng: random.Random,
    trials: int,
) -> tuple[int, ...] | None:
    """Find a randomized simple path whose vertex count lies in the range."""
    lo, hi = length_range
    for _ in range(trials):
        path = [start]
        used = {start}

        def search(v: int) -> bool:
            if v == end:
                return len(path) >= lo
            if len(path) >= hi:
                return False
            neighbors = list(language.graph.neighbors(v))
            rng.shuffle(neighbors)
            # Randomly postponing the target produces non-shortest paths too.
            for nxt in neighbors:
                if nxt in used or (nxt == end and len(path) + 1 < lo):
                    continue
                used.add(nxt)
                path.append(nxt)
                if search(nxt):
                    return True
                path.pop()
                used.remove(nxt)
            return False

        if search(start):
            return tuple(path)
    return None


def _encode_vertices(
    language: Language,
    vertices: Sequence[int],
    rng: random.Random,
    *,
    private_only: bool = False,
) -> tuple[str, tuple[int, ...]]:
    owners: dict[str, set[int]] = {}
    if private_only:
        for owner, book in enumerate(language.codebooks.books):
            for word in book:
                owners.setdefault(word, set()).add(owner)
    parts: list[str] = []
    cuts: list[int] = []
    total = 0
    for vertex in vertices:
        candidates = language.codebooks[vertex]
        if private_only:
            candidates = tuple(word for word in candidates if owners[word] == {vertex})
            if not candidates:
                raise ValueError(
                    f"vertex {vertex} has no private codeword for an unambiguous QA query"
                )
        word = rng.choice(candidates)
        parts.append(word)
        total += len(word)
        cuts.append(total)
    return "".join(parts), tuple(cuts)


def sample_qa(
    language: Language,
    task: TaskConfig,
    rng: random.Random,
    require_held_out: bool | None = None,
) -> QASample | None:
    """Draw one QA example; ``require_held_out`` filters consecutive query pairs."""
    length = rng.randint(*task.query_len)
    vertices = list(language.graph.vertices)
    if require_held_out:
        pair = rng.choice(task.held_out_pairs)
        if task.pair_direction == "unordered" and rng.randrange(2):
            pair = (pair[1], pair[0])
        pos = rng.randrange(length - 1)
        # Non-consecutive waypoint repetitions are allowed. Only an immediate
        # (v, v) pair is excluded because it does not define a path segment of
        # at least two vertices.
        for _ in range(task.path_trials):
            query = [rng.choice(vertices) for _ in range(length)]
            query[pos], query[pos + 1] = pair
            if all(u != v for u, v in zip(query, query[1:])):
                waypoints = tuple(query)
                break
        else:
            return None
    else:
        waypoints_list = [rng.choice(vertices)]
        for _ in range(1, length):
            candidates = [v for v in vertices if v != waypoints_list[-1]]
            waypoints_list.append(rng.choice(candidates))
        waypoints = tuple(waypoints_list)

    contains = query_contains_held_out(waypoints, task)
    if require_held_out is True and not contains:
        return None
    if require_held_out is False and contains:
        return None

    segments: list[tuple[int, ...]] = []
    for start, end in zip(waypoints, waypoints[1:]):
        segment = random_simple_path(
            language, start, end, task.segment_len, rng, task.path_trials
        )
        if segment is None:
            return None
        segments.append(segment)

    answer = list(segments[0])
    segment_cuts = [len(answer)]
    for segment in segments[1:]:
        answer.extend(segment[1:])
        segment_cuts.append(len(answer))
    # Query waypoints are not graph-adjacent in general. Shared codewords would
    # therefore make the question itself ambiguous under arbitrary overlap, so
    # prompts use each vertex's guaranteed private word. Answers may use every
    # assigned word and retain the original overlap regime.
    query_bits, query_cuts = _encode_vertices(
        language, waypoints, rng, private_only=True
    )
    answer_bits, answer_cuts = _encode_vertices(language, answer, rng)
    return QASample(
        query_bits=query_bits,
        answer_bits=answer_bits,
        query_vertices=waypoints,
        answer_walk=tuple(answer),
        query_cuts=query_cuts,
        answer_cuts=answer_cuts,
        segment_cuts=tuple(segment_cuts),
    )


def build_qa_splits(
    language: Language,
    task: TaskConfig,
    data: DataConfig,
    rng: random.Random,
    pool_tokens: int | None = None,
) -> Splits:
    """Build IID or held-out-pair QA splits to approximately the token budget."""
    budget = data.pool_tokens if pool_tokens is None else pool_tokens
    weights = data.split
    total_weight = sum(weights)
    budgets = [int(budget * weight / total_weight) for weight in weights]
    budgets[-1] += budget - sum(budgets)
    requirements = (
        (None, None, None)
        if task.split_mode == "iid"
        else (False, False, True)
    )
    built: list[tuple[QASample, ...]] = []
    seen: set[str] = set()
    for target, requirement in zip(budgets, requirements):
        items: list[QASample] = []
        tokens = 0
        attempts = 0
        max_attempts = max(10_000, target * 100)
        while tokens < target:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"could not fill QA split budget {target} after {attempts} attempts"
                )
            sample = sample_qa(language, task, rng, requirement)
            if sample is None or sample.bits in seen:
                continue
            seen.add(sample.bits)
            items.append(sample)
            tokens += len(sample.bits)
        built.append(tuple(items))
    return Splits(train=built[0], valid=built[1], test=built[2])


def qa_pool_stats(pool: Sequence[QASample]) -> dict[str, Any]:
    if not pool:
        return {"num_sentences": 0, "total_bits": 0}
    query_lengths = [len(sample.query_bits) for sample in pool]
    answer_lengths = [len(sample.answer_bits) for sample in pool]
    waypoint_counts = [len(sample.query_vertices) for sample in pool]
    answer_vertices = [len(sample.answer_walk) for sample in pool]
    return {
        "num_sentences": len(pool),
        "total_bits": sum(len(sample.bits) for sample in pool),
        "query_bits": _range_stats(query_lengths),
        "answer_bits": _range_stats(answer_lengths),
        "query_waypoints": _range_stats(waypoint_counts),
        "answer_vertices": _range_stats(answer_vertices),
    }


def _range_stats(values: Sequence[int]) -> dict[str, float | int]:
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
