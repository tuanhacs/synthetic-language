"""Codebook assignment: which codewords each vertex may speak.

A codebook assignment maps every vertex ``v`` to ``B_v`` (a set of ``k``
codewords drawn from the pool ``C^x``). v1 offers only ``disjoint-random``;
overlap constructions (harmonious colouring, sparse pairing, greedy maximal)
plug in as new strategies in :func:`assign` without touching anything else.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .codes.model import Code
from .config import LanguageConfig
from .graphs import GridGraph


@dataclass(frozen=True)
class CodebookReport:
    """Certification of the vertex-ambiguity condition (draft Theorem D.1)."""

    num_vertices: int
    k: int
    pairwise_disjoint: bool
    adjacent_disjoint: bool
    theorem_d1: bool
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_vertices": self.num_vertices,
            "k": self.k,
            "pairwise_disjoint": self.pairwise_disjoint,
            "adjacent_disjoint": self.adjacent_disjoint,
            "theorem_d1": self.theorem_d1,
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
    def k(self) -> int:
        """Codewords per vertex (uniform in v1)."""
        return len(self.books[0])

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

        edges = list(graph.directed_edges())
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
            k=self.k,
            pairwise_disjoint=pairwise_disjoint,
            adjacent_disjoint=adjacent_disjoint,
            theorem_d1=ok,
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
    graph: GridGraph, pool: Code, k: int, rng: random.Random
) -> Codebooks:
    """Sample ``|V| * k`` distinct codewords from the pool and deal ``k`` per vertex.

    The resulting codebooks are pairwise disjoint, which trivially satisfies the
    Theorem D.1 condition (both intersections are empty for distinct edges).
    """
    need = graph.num_vertices * k
    if need > len(pool):
        raise ValueError(f"pool has {len(pool)} words, need {need} for |V|*k")
    chosen = pool.sample_subset(need, rng)
    books = tuple(tuple(chosen.words[v * k : (v + 1) * k]) for v in range(graph.num_vertices))
    return Codebooks(books)


_STRATEGIES: dict[str, Callable[[GridGraph, Code, int, random.Random], Codebooks]] = {
    "disjoint-random": assign_disjoint_random,
}


def assign(
    cfg: LanguageConfig, graph: GridGraph, pool: Code, rng: random.Random
) -> Codebooks:
    """Dispatch to the assignment strategy named in the config."""
    try:
        strategy = _STRATEGIES[cfg.assignment]
    except KeyError:
        raise ValueError(
            f"unknown assignment {cfg.assignment!r} (known: {sorted(_STRATEGIES)})"
        ) from None
    return strategy(graph, pool, cfg.k, rng)


def available_assignments() -> Sequence[str]:
    return tuple(sorted(_STRATEGIES))
