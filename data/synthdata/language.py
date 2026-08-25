"""The language: sampler, general DP decoder, exact next-bit oracle, certifier.

A sentence is the concatenation (no separators) of one codeword per step of a
walk on the graph. The language is the set of *good strings* — strings that
encode at least one walk. The graph never appears in a sentence; it is the
hidden world a model must recover from bit statistics alone.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .codebooks import CodebookReport, Codebooks
from .codes.certify import CodeReport, certify as certify_code
from .graphs import GridGraph

#: Safety cap on the number of walks :meth:`Language.decode` returns.
MAX_WALKS = 100


@dataclass(frozen=True)
class Sample:
    """One sentence with its full ground truth.

    ``cuts[j]`` is the bit offset just *after* the codeword of step ``j``, so
    ``cuts`` has the same length as ``walk`` and ``cuts[-1] == len(bits)``.
    """

    bits: str
    walk: tuple[int, ...]
    cuts: tuple[int, ...]
    noised_bits: str | None = None

    @property
    def codewords(self) -> list[str]:
        out, start = [], 0
        for end in self.cuts:
            out.append(self.bits[start:end])
            start = end
        return out

    def reverse_walk(self) -> "Sample":
        """Reverse walk/codeword order without reversing any codeword's bits.

        This is intended as a paired-data transform before noise is applied:
        ``b1|...|bm`` becomes ``bm|...|b1`` and the walk is reversed with it.
        """
        if self.noised_bits is not None:
            raise ValueError("reverse_walk must be applied before noise")
        parts = list(reversed(self.codewords))
        cuts: list[int] = []
        total = 0
        for word in parts:
            total += len(word)
            cuts.append(total)
        return Sample(
            bits="".join(parts),
            walk=tuple(reversed(self.walk)),
            cuts=tuple(cuts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "noised_bits": self.noised_bits,
            "walk": list(self.walk),
            "cuts": list(self.cuts),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Sample":
        return Sample(
            bits=d["bits"],
            walk=tuple(int(v) for v in d["walk"]),
            cuts=tuple(int(c) for c in d["cuts"]),
            noised_bits=d.get("noised_bits"),
        )


@dataclass(frozen=True)
class LanguageReport:
    """Full certification of a language: code regime + vertex ambiguity."""

    code: CodeReport
    codebooks: CodebookReport
    regime: str
    unique_decoding: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.to_dict(),
            "codebooks": self.codebooks.to_dict(),
            "regime": self.regime,
            "unique_decoding": self.unique_decoding,
            "notes": list(self.notes),
        }


@dataclass
class Language:
    """A concrete language = graph + codebook assignment.

    All randomness is passed in explicitly, so ``(config, seed)`` fully
    determines every sample.
    """

    graph: GridGraph
    codebooks: Codebooks
    walk_len: tuple[int, int] = (8, 32)
    max_walks: int = MAX_WALKS

    # lookups built at init
    _words: list[tuple[str, ...]] = field(init=False, repr=False)
    _by_first_bit: list[tuple[tuple[str, ...], tuple[str, ...]]] = field(init=False, repr=False)
    _global_by_first_bit: tuple[tuple[str, ...], tuple[str, ...]] = field(init=False, repr=False)
    _max_word_len: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.codebooks) != self.graph.num_vertices:
            raise ValueError("codebooks and graph disagree on the number of vertices")
        lo, hi = self.walk_len
        if lo < 2 or hi < lo:
            raise ValueError(f"invalid walk_len {self.walk_len}")
        self._words = [tuple(self.codebooks[v]) for v in self.graph.vertices]
        self._by_first_bit = [
            (
                tuple(w for w in book if w[0] == "0"),
                tuple(w for w in book if w[0] == "1"),
            )
            for book in self._words
        ]
        global_words = self.codebooks.global_code().words
        self._global_by_first_bit = (
            tuple(word for word in global_words if word[0] == "0"),
            tuple(word for word in global_words if word[0] == "1"),
        )
        self._max_word_len = max(len(w) for book in self._words for w in book)

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #

    def sample_walk(self, walk_len: int, rng: random.Random) -> tuple[int, ...]:
        """Uniform random walk of ``walk_len`` vertices (uniform first vertex)."""
        v = rng.randrange(self.graph.num_vertices)
        walk = [v]
        for _ in range(walk_len - 1):
            v = rng.choice(self.graph.neighbors(v))
            walk.append(v)
        return tuple(walk)

    def encode(self, walk: Sequence[int], rng: random.Random) -> Sample:
        """Encode a walk, choosing one codeword per step uniformly from ``B_v``."""
        parts, cuts, total = [], [], 0
        for v in walk:
            word = rng.choice(self._words[v])
            parts.append(word)
            total += len(word)
            cuts.append(total)
        return Sample(bits="".join(parts), walk=tuple(walk), cuts=tuple(cuts))

    def sample(self, walk_len: int | None, rng: random.Random) -> Sample:
        """Draw one sentence. ``walk_len=None`` draws it uniformly from the range."""
        if walk_len is None:
            walk_len = rng.randint(*self.walk_len)
        return self.encode(self.sample_walk(walk_len, rng), rng)

    # ------------------------------------------------------------------ #
    # decoding (one general DP for every regime)
    # ------------------------------------------------------------------ #

    def is_decodable(self, bits: str) -> bool:
        """Whether ``bits`` can be fully segmented into global codewords.

        Vertex ownership and graph adjacency are intentionally ignored.  This
        is the decoding layer before the stricter grammatical-validity check.
        """
        if not bits or any(bit not in "01" for bit in bits):
            return False

        n = len(bits)
        reachable = bytearray(n + 1)
        reachable[0] = 1
        for i in range(n):
            if not reachable[i]:
                continue
            candidates = self._global_by_first_bit[0 if bits[i] == "0" else 1]
            for word in candidates:
                j = i + len(word)
                if j <= n and bits.startswith(word, i):
                    if j == n:
                        return True
                    reachable[j] = 1
        return False

    def _forward(self, bits: str) -> tuple[list[set[int]], dict[tuple[int, int], list[tuple[int, int]]]]:
        """Reachable states ``(cut position, vertex)`` plus backpointers.

        ``states[i]`` = vertices ``u`` such that ``bits[:i]`` is the concatenation
        of codewords along some walk ending at ``u``. Start states come from any
        vertex whose codebook has a prefix of ``bits``; transitions only go to
        graph neighbours. ``preds[(i, u)]`` lists the predecessor states, with
        ``(-1, -1)`` marking "walk starts here".
        """
        n = len(bits)
        states: list[set[int]] = [set() for _ in range(n + 1)]
        preds: dict[tuple[int, int], list[tuple[int, int]]] = {}

        for v in self.graph.vertices:  # initial states
            for word in self._words[v]:
                j = len(word)
                if j <= n and bits.startswith(word):
                    states[j].add(v)
                    preds.setdefault((j, v), []).append((-1, -1))

        for i in range(1, n):
            if not states[i]:
                continue
            for u in states[i]:
                for w in self.graph.neighbors(u):
                    for word in self._words[w]:
                        j = i + len(word)
                        if j <= n and bits.startswith(word, i):
                            states[j].add(w)
                            preds.setdefault((j, w), []).append((i, u))
        return states, preds

    def is_valid(self, bits: str) -> bool:
        """Is ``bits`` a good string (does it encode at least one walk)?

        Forward sweep with early exit as soon as a state reaches ``len(bits)``.
        No walk-length constraint is applied — validity is the language-theoretic
        notion from the draft (any walk with at least 2 vertices).
        """
        n = len(bits)
        if n == 0:
            return False
        frontier: list[set[int]] = [set() for _ in range(n + 1)]
        for v in self.graph.vertices:
            for word in self._words[v]:
                j = len(word)
                if j <= n and bits.startswith(word):
                    frontier[j].add(v)
        for i in range(1, n + 1):
            if not frontier[i]:
                continue
            if i == n:
                return True  # the whole string parsed
            for u in frontier[i]:
                for w in self.graph.neighbors(u):
                    for word in self._words[w]:
                        j = i + len(word)
                        if j <= n and bits.startswith(word, i):
                            frontier[j].add(w)
        return False

    def decode(self, bits: str) -> list[list[int]]:
        """All walks consistent with ``bits`` (empty list if the string is invalid).

        In a certified regime (UD code + Theorem D.1 satisfied) the result is a
        single walk. The number of returned walks is capped at
        :attr:`max_walks` to stay safe on pathological inputs.
        """
        if not bits:
            return []
        n = len(bits)
        states, preds = self._forward(bits)
        if not states[n]:
            return []

        walks: list[list[int]] = []

        def backtrack(i: int, u: int, tail: list[int]) -> None:
            if len(walks) >= self.max_walks:
                return
            tail = [u] + tail
            for j, prev in preds.get((i, u), ()):
                if j == -1:
                    walks.append(list(tail))
                    if len(walks) >= self.max_walks:
                        return
                else:
                    backtrack(j, prev, tail)
                    if len(walks) >= self.max_walks:
                        return

        for u in sorted(states[n]):
            backtrack(n, u, [])
            if len(walks) >= self.max_walks:
                break
        return walks

    def segment(self, bits: str, walk: Sequence[int]) -> list[str] | None:
        """Split ``bits`` into the codewords spoken along ``walk``, or ``None``.

        Not greedy: with a non-prefix-free code several codewords of ``B_v`` can
        match at a position and only one continuation works out, so this
        backtracks over the choices.
        """

        def rec(pos: int, step: int) -> list[str] | None:
            if step == len(walk):
                return [] if pos == len(bits) else None
            for word in self._words[walk[step]]:
                if bits.startswith(word, pos):
                    tail = rec(pos + len(word), step + 1)
                    if tail is not None:
                        return [word] + tail
            return None

        return rec(0, 0)

    def decode_unique(self, bits: str) -> list[int] | None:
        """The walk, if ``bits`` decodes to exactly one; else ``None``."""
        walks = self.decode(bits)
        return walks[0] if len(walks) == 1 else None

    # ------------------------------------------------------------------ #
    # exact next-bit oracle
    # ------------------------------------------------------------------ #
    #
    # The sampler is a Markov process over states (step index, vertex, offset
    # inside the current codeword). Running the parse forward with the sampler's
    # own probabilities therefore yields the exact Bayes-optimal next-token
    # distribution, hence the cross-entropy floor of the language.
    #
    # Weights below deliberately exclude the walk-length factor: a parse that
    # has *started* j codewords is compatible with any drawn length m >= j, and
    # because the graph has no dead ends every such m admits a completion whose
    # remaining choices sum to probability 1. So the walk length only enters as
    # P(m >= j) for continuations and P(m = j) for stopping.

    def _walk_len_factors(self) -> tuple[int, int, float]:
        lo, hi = self.walk_len
        return lo, hi, 1.0 / (hi - lo + 1)

    def _p_at_least(self, j: int) -> float:
        lo, hi, p = self._walk_len_factors()
        if j <= lo:
            return 1.0
        if j > hi:
            return 0.0
        return (hi - j + 1) * p

    def _p_exactly(self, j: int) -> float:
        lo, hi, p = self._walk_len_factors()
        return p if lo <= j <= hi else 0.0

    def _start(
        self, bit: str
    ) -> tuple[dict[tuple[int, int, str, int], float], dict[tuple[int, int], float]]:
        """Weighted (partial, complete) state sets after the very first bit."""
        partial: dict[tuple[int, int, str, int], float] = {}
        complete: dict[tuple[int, int], float] = {}
        p_v = 1.0 / self.graph.num_vertices
        for v in self.graph.vertices:
            book = self._words[v]
            wt = p_v / len(book)
            for word in self._by_first_bit[v][0 if bit == "0" else 1]:
                if len(word) == 1:
                    complete[(1, v)] = complete.get((1, v), 0.0) + wt
                else:
                    partial[(1, v, word, 1)] = partial.get((1, v, word, 1), 0.0) + wt
        return partial, complete

    def _step(
        self,
        partial: dict[tuple[int, int, str, int], float],
        complete: dict[tuple[int, int], float],
        bit: str,
    ) -> tuple[dict[tuple[int, int, str, int], float], dict[tuple[int, int], float]]:
        """Consume one bit from the weighted state set.

        ``partial[(steps, vertex, word, offset)]``: ``steps`` codewords started,
        the last one (``word`` at ``vertex``) has ``offset`` bits consumed.
        ``complete[(steps, vertex)]``: ``steps`` codewords fully consumed.
        """
        new_partial: dict[tuple[int, int, str, int], float] = {}
        new_complete: dict[tuple[int, int], float] = {}
        idx = 0 if bit == "0" else 1

        for (steps, v, word, off), wt in partial.items():
            if word[off] != bit:
                continue
            if off + 1 == len(word):
                key = (steps, v)
                new_complete[key] = new_complete.get(key, 0.0) + wt
            else:
                key2 = (steps, v, word, off + 1)
                new_partial[key2] = new_partial.get(key2, 0.0) + wt

        _, hi, _ = self._walk_len_factors()
        for (steps, u), wt in complete.items():
            if steps + 1 > hi:
                continue  # cannot start another codeword
            p_nb = 1.0 / self.graph.degree(u)
            for w in self.graph.neighbors(u):
                book = self._words[w]
                p_b = wt * p_nb / len(book)
                for word in self._by_first_bit[w][idx]:
                    if len(word) == 1:
                        key = (steps + 1, w)
                        new_complete[key] = new_complete.get(key, 0.0) + p_b
                    else:
                        key2 = (steps + 1, w, word, 1)
                        new_partial[key2] = new_partial.get(key2, 0.0) + p_b
        return new_partial, new_complete

    def _mass(
        self,
        partial: dict[tuple[int, int, str, int], float],
        complete: dict[tuple[int, int], float],
    ) -> float:
        """Probability that a sampled sentence starts with the consumed prefix."""
        total = 0.0
        for (steps, _v, _w, _o), wt in partial.items():
            total += wt * self._p_at_least(steps)
        for (steps, _v), wt in complete.items():
            total += wt * self._p_at_least(steps)
        return total

    def _eos_mass(self, complete: dict[tuple[int, int], float]) -> float:
        """Probability that the sentence equals the consumed prefix exactly."""
        return sum(wt * self._p_exactly(steps) for (steps, _v), wt in complete.items())

    def next_bit_dist(self, prefix: str) -> dict[str, float]:
        """Exact optimal next-token distribution over ``{'0', '1', 'EOS'}``.

        Matches the sampler's probability model: walk length uniform in
        ``walk_len``, first vertex uniform, next vertex uniform among neighbours,
        codeword uniform in ``B_v``. Returns an all-zero dict for a prefix that
        no sentence of the language starts with.
        """
        partial: dict[tuple[int, int, str, int], float] = {}
        complete: dict[tuple[int, int], float] = {}
        for i, bit in enumerate(prefix):
            partial, complete = (
                self._start(bit) if i == 0 else self._step(partial, complete, bit)
            )
            if not partial and not complete:
                return {"0": 0.0, "1": 0.0, "EOS": 0.0}

        masses: dict[str, float] = {}
        for bit in ("0", "1"):
            np_, nc_ = self._start(bit) if not prefix else self._step(partial, complete, bit)
            masses[bit] = self._mass(np_, nc_)
        masses["EOS"] = self._eos_mass(complete)

        total = sum(masses.values())
        if total <= 0.0:
            return {"0": 0.0, "1": 0.0, "EOS": 0.0}
        return {key: val / total for key, val in masses.items()}

    def token_logprobs(self, bits: str) -> list[float]:
        """log2 probabilities of every token of ``BOS -> bits -> EOS``.

        One forward pass: ``len(bits) + 1`` values (each bit, then EOS).
        """
        out: list[float] = []
        partial: dict[tuple[int, int, str, int], float] = {}
        complete: dict[tuple[int, int], float] = {}

        for i, bit in enumerate(bits):
            other = "1" if bit == "0" else "0"
            if i == 0:
                partial, complete = self._start(bit)
                mass_this = self._mass(partial, complete)
                total = mass_this + self._mass(*self._start(other))
            else:
                np_, nc_ = self._step(partial, complete, bit)
                mass_this = self._mass(np_, nc_)
                total = (
                    mass_this
                    + self._mass(*self._step(partial, complete, other))
                    + self._eos_mass(complete)
                )
                partial, complete = np_, nc_
            if mass_this <= 0.0 or total <= 0.0:
                raise ValueError(f"bits are not a valid prefix of the language at position {i}")
            out.append(math.log2(mass_this / total))

        eos = self._eos_mass(complete)
        total = (
            eos
            + self._mass(*self._step(partial, complete, "0"))
            + self._mass(*self._step(partial, complete, "1"))
        )
        if eos <= 0.0 or total <= 0.0:
            raise ValueError("string cannot terminate here (walk length out of range?)")
        out.append(math.log2(eos / total))
        return out

    def entropy_floor(self, samples: Iterable[Sample | str]) -> float:
        """Bayes-optimal cross-entropy in bits/token on held-out samples.

        Averaged over all tokens of ``bits + EOS`` (BOS is never predicted).
        This is the loss floor a perfect model would reach on this language.
        """
        total_bits, total_tokens = 0.0, 0
        for item in samples:
            bits = item if isinstance(item, str) else item.bits
            lps = self.token_logprobs(bits)
            total_bits -= sum(lps)
            total_tokens += len(lps)
        if total_tokens == 0:
            raise ValueError("entropy_floor needs at least one sample")
        return total_bits / total_tokens

    # ------------------------------------------------------------------ #
    # certification
    # ------------------------------------------------------------------ #

    def certify(self, declared_type: str | None = None) -> LanguageReport:
        """Certify the language: code regime + vertex-ambiguity condition."""
        code_report = certify_code(self.codebooks.global_code())
        book_report = self.codebooks.certify(self.graph)
        unique = code_report.uniquely_decipherable and book_report.theorem_d1
        notes: list[str] = []
        if not code_report.uniquely_decipherable:
            notes.append("global code is not UD: strings may admit several segmentations")
        if not book_report.theorem_d1:
            notes.append("Theorem D.1 condition violated: some good strings have several walks")
        if declared_type is not None and not code_report.matches(declared_type):
            notes.append(
                f"declared code type {declared_type!r} does not match certified regime "
                f"{code_report.regime!r}"
            )
        return LanguageReport(
            code=code_report,
            codebooks=book_report,
            regime=code_report.regime,
            unique_decoding=unique,
            notes=tuple(notes),
        )
