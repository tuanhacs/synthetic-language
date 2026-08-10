"""The :class:`Code` value object: a finite set of non-empty binary words."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True)
class Code:
    """A finite code: an ordered tuple of distinct non-empty binary words.

    The order is part of the object (it makes every downstream random draw
    reproducible), but the mathematical object is the underlying set.
    """

    words: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.words:
            raise ValueError("a code must contain at least one codeword")
        if len(set(self.words)) != len(self.words):
            raise ValueError("codewords must be distinct")
        for w in self.words:
            if not w or any(ch not in "01" for ch in w):
                raise ValueError(f"invalid codeword {w!r}: must be a non-empty string of 0/1")

    # ---- container protocol ----------------------------------------------- #

    def __len__(self) -> int:
        return len(self.words)

    def __iter__(self) -> Iterator[str]:
        return iter(self.words)

    def __contains__(self, word: object) -> bool:
        return word in self.words

    def __getitem__(self, i: int) -> str:
        return self.words[i]

    # ---- statistics ------------------------------------------------------- #

    @property
    def min_len(self) -> int:
        return min(len(w) for w in self.words)

    @property
    def max_len(self) -> int:
        return max(len(w) for w in self.words)

    @property
    def mean_len(self) -> float:
        return sum(len(w) for w in self.words) / len(self.words)

    def length_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for w in self.words:
            hist[len(w)] = hist.get(len(w), 0) + 1
        return dict(sorted(hist.items()))

    # ---- constructions ---------------------------------------------------- #

    def power(self, x: int) -> "Code":
        """``C^x``: all concatenations of exactly ``x`` codewords.

        If ``C`` is prefix-free / suffix-free / UD then so is ``C^x``
        (draft Prop. E.2 and the analogous induction on the first block), so the
        powered pool inherits the certified regime of the base code.
        """
        if x < 1:
            raise ValueError("power x must be >= 1")
        if x == 1:
            return Code(self.words)
        words = ["".join(combo) for combo in itertools.product(self.words, repeat=x)]
        return Code(tuple(words))

    def sample_subset(self, size: int, rng: random.Random) -> "Code":
        """Sample ``size`` distinct codewords uniformly without replacement."""
        if size > len(self.words):
            raise ValueError(f"cannot sample {size} words from a code of size {len(self.words)}")
        return Code(tuple(rng.sample(self.words, size)))

    def reversed_code(self) -> "Code":
        return Code(tuple(w[::-1] for w in self.words))

    # ---- serialisation ---------------------------------------------------- #

    def to_list(self) -> list[str]:
        return list(self.words)

    @staticmethod
    def from_list(words: Sequence[str]) -> "Code":
        return Code(tuple(words))


def load_codeword_pool(path: str | Path) -> Code:
    """Load an ordered codeword pool from a JSON array.

    The canonical format is simply ``["001", "101", ...]``. An object with a
    ``codewords`` array is also accepted for callers that want to attach their
    own metadata. Array order is preserved because seeded random assignment is
    defined relative to that order.
    """
    source = Path(path)
    try:
        # utf-8-sig accepts ordinary UTF-8 and files with the BOM commonly
        # emitted by Windows tools such as PowerShell and Excel.
        doc = json.loads(source.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ValueError(f"codeword pool file does not exist: {source}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in codeword pool file {source}: {exc}") from exc

    words = doc.get("codewords") if isinstance(doc, dict) else doc
    if not isinstance(words, list):
        raise ValueError(
            f"codeword pool file {source} must be a JSON array or an object with 'codewords'"
        )
    if any(not isinstance(word, str) for word in words):
        raise ValueError("every codeword in the pool must be a JSON string")
    try:
        return Code(tuple(words))
    except ValueError as exc:
        raise ValueError(f"invalid codeword pool {source}: {exc}") from exc
