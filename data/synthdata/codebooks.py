"""Codebook assignment: which codewords each vertex may speak.

A codebook assignment maps every vertex ``v`` to ``B_v`` (a non-empty set of
codewords drawn from the pool ``C^x``). ``disjoint-random`` partitions a sampled
subset; ``arbitrary-overlap`` constructs codeword supports satisfying the exact
graph-relative unique-decoding condition from the draft.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .codes.certify import certify as certify_code
from .codes.model import Code
from .config import LanguageConfig
from .graphs import GridGraph


@dataclass(frozen=True)
class CodebookReport:
    """Certification of the vertex-ambiguity condition (draft Theorem D.1)."""

    num_vertices: int
    k_per_vertex: tuple[int, ...]
    pairwise_disjoint: bool
    adjacent_disjoint: bool
    theorem_d1: bool
    shared_codewords: int
    overlap_pairs: int
    max_pairwise_overlap: int
    max_support_size: int
    support_size_histogram: tuple[tuple[int, int], ...]
    max_support_edge_multiplicity: int
    violations: tuple[str, ...] = ()

    @property
    def k_range(self) -> tuple[int, int]:
        return min(self.k_per_vertex), max(self.k_per_vertex)

    @property
    def k(self) -> int | tuple[int, int]:
        lo, hi = self.k_range
        return lo if lo == hi else (lo, hi)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_vertices": self.num_vertices,
            "k": self.k if isinstance(self.k, int) else list(self.k),
            "k_per_vertex": list(self.k_per_vertex),
            "total_assignments": sum(self.k_per_vertex),
            "pairwise_disjoint": self.pairwise_disjoint,
            "adjacent_disjoint": self.adjacent_disjoint,
            "theorem_d1": self.theorem_d1,
            "shared_codewords": self.shared_codewords,
            "overlap_pairs": self.overlap_pairs,
            "max_pairwise_overlap": self.max_pairwise_overlap,
            "max_support_size": self.max_support_size,
            "support_size_histogram": {
                str(size): count for size, count in self.support_size_histogram
            },
            "max_support_edge_multiplicity": self.max_support_edge_multiplicity,
            "violations": list(self.violations),
        }


@dataclass(frozen=True)
class Codebooks:
    """Immutable mapping vertex -> tuple of codewords."""

    books: tuple[tuple[str, ...], ...]  # indexed by vertex id

    def __post_init__(self) -> None:
        for v, book in enumerate(self.books):
            if not book:
                raise ValueError(f"vertex {v} has an empty codebook")

    def __len__(self) -> int:
        return len(self.books)

    def __getitem__(self, v: int) -> tuple[str, ...]:
        return self.books[v]

    @property
    def k_per_vertex(self) -> tuple[int, ...]:
        return tuple(len(book) for book in self.books)

    @property
    def k_range(self) -> tuple[int, int]:
        sizes = self.k_per_vertex
        return min(sizes), max(sizes)

    @property
    def k(self) -> int | tuple[int, int]:
        """Fixed ``k``, or the actual ``(min, max)`` for variable-size books."""
        lo, hi = self.k_range
        return lo if lo == hi else (lo, hi)

    def all_words(self) -> list[str]:
        return [w for book in self.books for w in book]

    def global_code(self) -> Code:
        """``C = union of B_v`` — the global code actually used by the language."""
        seen: dict[str, None] = {}
        for word in self.all_words():
            seen.setdefault(word, None)
        return Code(tuple(seen))

    # ---- certification ---------------------------------------------------- #

    def certify(self, graph: GridGraph, max_violations: int = 10) -> CodebookReport:
        """Check the unique-walk condition over all pairs of distinct directed edges.

        Theorem D.1: with ``C`` UD, every good string determines its walk uniquely
        iff for every two distinct directed edges ``(u, v) != (u', v')``:
        ``B_u n B_u' = empty`` or ``B_v n B_v' = empty``.
        """
        sets = [set(book) for book in self.books]
        violations: list[str] = []

        pairwise_disjoint = True
        for u in range(len(self.books)):
            for w in range(u + 1, len(self.books)):
                if sets[u] & sets[w]:
                    pairwise_disjoint = False
                    break
            if not pairwise_disjoint:
                break

        adjacent_disjoint = all(
            not (sets[u] & sets[v]) for u, v in graph.directed_edges()
        )

        word_supports: dict[str, set[int]] = {}
        for v, book in enumerate(self.books):
            for word in book:
                word_supports.setdefault(word, set()).add(v)
        supports = list(word_supports.values())
        support_histogram = Counter(len(support) for support in supports)
        overlap_sizes = [
            len(sets[u] & sets[v])
            for u in range(len(sets))
            for v in range(u + 1, len(sets))
        ]

        edges = list(graph.directed_edges())
        edge_set = set(edges)
        max_support_edge_multiplicity = max(
            (
                sum((u, v) in edge_set for u in left for v in right)
                for left in supports
                for right in supports
            ),
            default=0,
        )
        ok = True
        for i, (u, v) in enumerate(edges):
            for u2, v2 in edges[i + 1 :]:
                if sets[u] & sets[u2] and sets[v] & sets[v2]:
                    ok = False
                    if len(violations) < max_violations:
                        violations.append(f"({u},{v}) vs ({u2},{v2})")
            if not ok and len(violations) >= max_violations:
                break

        return CodebookReport(
            num_vertices=len(self.books),
            k_per_vertex=self.k_per_vertex,
            pairwise_disjoint=pairwise_disjoint,
            adjacent_disjoint=adjacent_disjoint,
            theorem_d1=ok,
            shared_codewords=sum(len(support) > 1 for support in supports),
            overlap_pairs=sum(size > 0 for size in overlap_sizes),
            max_pairwise_overlap=max(overlap_sizes, default=0),
            max_support_size=max((len(support) for support in supports), default=0),
            support_size_histogram=tuple(sorted(support_histogram.items())),
            max_support_edge_multiplicity=max_support_edge_multiplicity,
            violations=tuple(violations),
        )

    # ---- serialisation ---------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {"books": [list(book) for book in self.books]}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Codebooks":
        return Codebooks(tuple(tuple(str(w) for w in book) for book in d["books"]))


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #


def assign_disjoint_random(
    graph: GridGraph, pool: Code, k: int | tuple[int, int], rng: random.Random
) -> Codebooks:
    """Sample distinct words and deal a fixed or random count per vertex.

    The resulting codebooks are pairwise disjoint, which trivially satisfies the
    Theorem D.1 condition. For ``k=(lo, hi)``, every vertex count is sampled
    independently and uniformly from the inclusive range using ``rng``.
    """
    if isinstance(k, int):
        counts = [k] * graph.num_vertices
    else:
        lo, hi = k
        counts = [rng.randint(lo, hi) for _ in graph.vertices]
    need = sum(counts)
    if need > len(pool):
        raise ValueError(f"pool has {len(pool)} words, assignment sampled {need}")
    chosen = pool.sample_subset(need, rng)
    books: list[tuple[str, ...]] = []
    offset = 0
    for count in counts:
        books.append(tuple(chosen.words[offset : offset + count]))
        offset += count
    return Codebooks(tuple(books))


def _support_pair_edge_count(
    left: frozenset[int], right: frozenset[int], directed_edges: set[tuple[int, int]]
) -> int:
    return sum((u, v) in directed_edges for u in left for v in right)


def _support_is_admissible(
    candidate: frozenset[int],
    supports: Sequence[frozenset[int]],
    directed_edges: set[tuple[int, int]],
) -> bool:
    """Exact support-set condition from the draft, incremental form."""
    if candidate in supports:
        return False  # spread overlap instead of stacking identical supports
    if _support_pair_edge_count(candidate, candidate, directed_edges) > 1:
        return False
    return all(
        _support_pair_edge_count(candidate, support, directed_edges) <= 1
        and _support_pair_edge_count(support, candidate, directed_edges) <= 1
        for support in supports
    )


def _regime_subset(
    pool: Code, size: int, code_type: str, rng: random.Random, max_tries: int
) -> Code:
    """Select ``size`` words while preserving the declared exact code regime.

    Suffix and hard-UD regimes need explicit non-prefix/non-suffix witnesses;
    otherwise a small random subset can accidentally become a stronger code.
    Every proposed subset is still certified exactly before it is returned.
    """
    if size > len(pool):
        raise ValueError(f"pool has {len(pool)} words but assignment needs {size}")
    words = pool.words

    def prefix_witnesses(items: Sequence[str]) -> list[tuple[str, str]]:
        ordered = sorted(items)
        return [
            (short, ordered[i + 1])
            for i, short in enumerate(ordered[:-1])
            if ordered[i + 1].startswith(short)
        ]

    prefix_pairs = (
        prefix_witnesses(words) if code_type in ("suffix-free", "ud") else []
    )
    suffix_pairs = (
        [(short_rev[::-1], long_rev[::-1]) for short_rev, long_rev in
         prefix_witnesses(tuple(word[::-1] for word in words))]
        if code_type == "ud"
        else []
    )
    if code_type in ("suffix-free", "ud") and not prefix_pairs:
        raise ValueError(
            f"pool cannot realise exact {code_type!r}: it has no prefix-relation witness"
        )
    if code_type == "ud" and not suffix_pairs:
        raise ValueError("pool cannot realise exact 'ud': it has no suffix-relation witness")

    for _ in range(max_tries):
        mandatory: list[str] = []
        if code_type in ("suffix-free", "ud"):
            mandatory.extend(rng.choice(prefix_pairs))
        if code_type == "ud":
            mandatory.extend(rng.choice(suffix_pairs))
        mandatory = list(dict.fromkeys(mandatory))
        if len(mandatory) > size:
            continue
        mandatory_set = set(mandatory)
        available = [word for word in words if word not in mandatory_set]
        selected = mandatory + rng.sample(available, size - len(mandatory))
        rng.shuffle(selected)
        code = Code(tuple(selected))
        if certify_code(code).matches(code_type):
            return code
    raise ValueError(
        f"could not select {size} codewords with exact regime {code_type!r} "
        f"within {max_tries} tries"
    )


def assign_arbitrary_overlap(
    cfg: LanguageConfig, graph: GridGraph, pool: Code, rng: random.Random
) -> Codebooks:
    """Greedy arbitrary-overlap construction using codeword support sets.

    One singleton support anchors every vertex. Shared supports are then added
    while preserving ``|E_dir intersect (A_j x A_k)| <= 1`` for all support
    pairs. Remaining target slots receive private singleton supports.
    """
    overlap = cfg.overlap
    if overlap is None:
        raise ValueError("arbitrary-overlap assignment requires overlap config")
    if isinstance(cfg.k, int):
        targets = [cfg.k] * graph.num_vertices
    else:
        lo, hi = cfg.k
        targets = [rng.randint(lo, hi) for _ in graph.vertices]

    support_lo, support_hi = overlap.support_size
    residual_total = sum(target - 1 for target in targets)
    minimum_shared_slots = overlap.shared_codewords * support_lo
    if residual_total < minimum_shared_slots:
        raise ValueError(
            f"overlap target needs at least {minimum_shared_slots} shared slots, "
            f"but sampled vertex capacities leave only {residual_total}"
        )

    directed_edges = set(graph.directed_edges())
    accepted: list[frozenset[int]] | None = None
    for _ in range(overlap.max_restarts):
        supports = [frozenset((v,)) for v in graph.vertices]
        remaining = [target - 1 for target in targets]
        success = True
        for shared_idx in range(overlap.shared_codewords):
            remaining_words = overlap.shared_codewords - shared_idx - 1
            added = False
            for _ in range(overlap.candidate_trials):
                eligible = [v for v in graph.vertices if remaining[v] > 0]
                max_size = min(
                    support_hi,
                    len(eligible),
                    sum(remaining) - remaining_words * support_lo,
                )
                if max_size < support_lo:
                    break
                size = rng.randint(support_lo, max_size)
                candidate = frozenset(rng.sample(eligible, size))
                if not _support_is_admissible(candidate, supports, directed_edges):
                    continue
                supports.append(candidate)
                for v in candidate:
                    remaining[v] -= 1
                added = True
                break
            if not added:
                success = False
                break
        if success:
            for v, count in enumerate(remaining):
                supports.extend(frozenset((v,)) for _ in range(count))
            accepted = supports
            break
    if accepted is None:
        raise ValueError(
            f"could not construct {overlap.shared_codewords} admissible shared supports "
            f"after {overlap.max_restarts} restarts x {overlap.candidate_trials} trials"
        )

    chosen = _regime_subset(
        pool, len(accepted), cfg.code.type, rng, overlap.max_restarts
    )
    books: list[list[str]] = [[] for _ in graph.vertices]
    for word, support in zip(chosen.words, accepted):
        for v in support:
            books[v].append(word)
    result = Codebooks(tuple(tuple(book) for book in books))
    if result.k_per_vertex != tuple(targets):
        raise RuntimeError("arbitrary-overlap construction did not meet vertex targets")
    return result


def assign(
    cfg: LanguageConfig, graph: GridGraph, pool: Code, rng: random.Random
) -> Codebooks:
    """Dispatch to the assignment strategy named in the config."""
    if cfg.assignment == "disjoint-random":
        return assign_disjoint_random(graph, pool, cfg.k, rng)
    if cfg.assignment == "arbitrary-overlap":
        return assign_arbitrary_overlap(cfg, graph, pool, rng)
    raise ValueError(f"unknown assignment {cfg.assignment!r}")


def available_assignments() -> Sequence[str]:
    return ("arbitrary-overlap", "disjoint-random")
