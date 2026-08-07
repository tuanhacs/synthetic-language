"""Code certification: prefix-free, suffix-free, and unique decipherability.

The certifier checks the *artifact* (the generated code), not the algorithm on
paper. Its report is stored in the dataset manifest, so every dataset carries
its own proof of the segmentation regime it claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .model import Code


def is_prefix_free(code: Iterable[str]) -> bool:
    """True iff no codeword is a proper prefix of another (an antichain)."""
    words = sorted(code)
    for i, w in enumerate(words):
        for other in words[i + 1 :]:
            if not other.startswith(w):
                break  # sorted order: no later word can start with w either
            if other != w:
                return False
    return True


def is_suffix_free(code: Iterable[str]) -> bool:
    """True iff no codeword is a proper suffix of another."""
    return is_prefix_free(w[::-1] for w in code)


def _left_quotient(a: Iterable[str], b: Iterable[str]) -> set[str]:
    """``a^{-1} b`` = {t : u t = v for some u in a, v in b, t non-empty}."""
    b = list(b)
    out: set[str] = set()
    for u in a:
        for v in b:
            if len(v) > len(u) and v.startswith(u):
                out.add(v[len(u) :])
    return out


def sardinas_patterson(code: Iterable[str]) -> bool:
    """Decide unique decipherability with the Sardinas-Patterson algorithm.

    ``S_1 = C^{-1}C \\ {eps}``; ``S_{k+1} = C^{-1} S_k  u  S_k^{-1} C``.
    The code is UD iff no ``S_k`` ever contains a codeword. The iteration is
    finite: every dangling suffix is a suffix of some codeword, so the state
    space is finite and we stop when ``S_k`` is empty or repeats a seen set.
    """
    words = list(code)
    if len(set(words)) != len(words):
        return False  # repeated codeword: trivially ambiguous
    cset = set(words)

    s = _left_quotient(cset, cset)  # dangling suffixes, epsilon excluded by construction
    seen: set[frozenset[str]] = set()
    while s:
        if s & cset:
            return False
        key = frozenset(s)
        if key in seen:
            return True  # cycle without hitting C
        seen.add(key)
        s = _left_quotient(cset, s) | _left_quotient(s, cset)
    return True


@dataclass(frozen=True)
class CodeReport:
    """Result of running the full certification suite on one code."""

    size: int
    min_len: int
    max_len: int
    prefix_free: bool
    suffix_free: bool
    uniquely_decipherable: bool

    @property
    def regime(self) -> str:
        """Coarsest label describing the segmentation regime of the code."""
        if not self.uniquely_decipherable:
            return "non-UD"
        if self.prefix_free and self.suffix_free:
            return "prefix-free+suffix-free"
        if self.prefix_free:
            return "prefix-free"
        if self.suffix_free:
            return "suffix-free"
        return "ud"

    def matches(self, code_type: str) -> bool:
        """Does this report certify the requested code type?

        ``ud`` demands UD but *neither* prefix-free nor suffix-free, so that the
        three code types are genuinely distinct difficulty levels.
        """
        if not self.uniquely_decipherable:
            return False
        if code_type == "prefix-free":
            return self.prefix_free
        if code_type == "suffix-free":
            return self.suffix_free and not self.prefix_free
        if code_type == "ud":
            return not self.prefix_free and not self.suffix_free
        raise ValueError(f"unknown code type {code_type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "min_len": self.min_len,
            "max_len": self.max_len,
            "prefix_free": self.prefix_free,
            "suffix_free": self.suffix_free,
            "uniquely_decipherable": self.uniquely_decipherable,
            "regime": self.regime,
        }


def certify(code: Code) -> CodeReport:
    """Run all three checks on ``code``."""
    return CodeReport(
        size=len(code),
        min_len=code.min_len,
        max_len=code.max_len,
        prefix_free=is_prefix_free(code),
        suffix_free=is_suffix_free(code),
        uniquely_decipherable=sardinas_patterson(code),
    )
