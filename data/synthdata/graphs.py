"""Graphs: the hidden worlds. v1 supports square grid graphs only.

Vertices are integers ``0 .. n*n-1`` in row-major order: vertex ``v`` sits at
row ``v // n``, column ``v % n``. Edges are the 4-neighbourhood, undirected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class GridGraph:
    """Undirected n x n grid graph with pre-computed adjacency."""

    n: int
    adj: tuple[tuple[int, ...], ...]

    # ---- construction ----------------------------------------------------- #

    @property
    def num_vertices(self) -> int:
        return self.n * self.n

    @property
    def vertices(self) -> range:
        return range(self.num_vertices)

    def neighbors(self, v: int) -> tuple[int, ...]:
        return self.adj[v]

    def degree(self, v: int) -> int:
        return len(self.adj[v])

    def is_edge(self, u: int, v: int) -> bool:
        return v in self.adj[u]

    def directed_edges(self) -> Iterator[tuple[int, int]]:
        """Every ordered pair (u, v) with u ~ v."""
        for u in self.vertices:
            for v in self.adj[u]:
                yield u, v

    def num_edges(self) -> int:
        return sum(len(a) for a in self.adj) // 2

    def name(self) -> str:
        return f"grid-{self.n}x{self.n}"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "grid", "n": self.n, "adj": [list(a) for a in self.adj]}


def make_grid(n: int) -> GridGraph:
    """Build the undirected n x n grid graph."""
    if n < 2:
        raise ValueError("grid side length must be >= 2")
    adj: list[tuple[int, ...]] = []
    for v in range(n * n):
        r, c = divmod(v, n)
        nb = []
        if r > 0:
            nb.append(v - n)
        if r < n - 1:
            nb.append(v + n)
        if c > 0:
            nb.append(v - 1)
        if c < n - 1:
            nb.append(v + 1)
        adj.append(tuple(sorted(nb)))
    return GridGraph(n=n, adj=tuple(adj))


def graph_from_dict(d: dict[str, Any]) -> GridGraph:
    """Inverse of :meth:`GridGraph.to_dict`."""
    if d.get("type") != "grid":
        raise ValueError(f"unsupported graph type: {d.get('type')!r}")
    return GridGraph(n=int(d["n"]), adj=tuple(tuple(int(x) for x in a) for a in d["adj"]))


def graph_from_spec(spec: str) -> GridGraph:
    """Build a graph from a config spec string such as ``'grid-4x4'``."""
    from .config import LanguageConfig  # local import to avoid a cycle

    return make_grid(LanguageConfig(graph=spec).grid_n)
