"""Base-code generators, one per code type.

This is the *only* part of the pipeline that branches on the code type. All
three generators produce a small base code (``base_size`` words) with the
requested property; the shared pipeline then takes ``C^x`` as the codeword pool
(design option A), which keeps codeword length distribution and internal block
structure identical across code types.

Every generator takes an explicit ``random.Random`` — no global random state.
"""

from __future__ import annotations

import random

from ..config import CodeConfig
from .certify import is_prefix_free, is_suffix_free, sardinas_patterson
from .model import Code

#: Known UD code that is neither prefix-free nor suffix-free (draft running example).
UD_FALLBACK = ("0", "01", "110")


class CodeGenerationError(RuntimeError):
    """Raised when a generator cannot meet its constraints within max_tries."""


def make_prefix_free(
    base_size: int,
    depth_range: tuple[int, int],
    rng: random.Random,
    max_tries: int = 10_000,
) -> Code:
    """Prefix-free code by random binary-tree pruning.

    Repeatedly pick a leaf of the current tree and either keep it as a codeword
    or expand it into its two children, until ``base_size`` leaves remain, all
    with depth inside ``depth_range``. The leaves of a pruned binary tree form
    an antichain, hence the code is prefix-free by construction.
    """
    dmin, dmax = depth_range
    if 2**dmax < base_size:
        raise CodeGenerationError(
            f"cannot fit {base_size} leaves within depth {dmax} (max {2 ** dmax})"
        )
    for _ in range(max_tries):
        leaves: list[str] = [""]  # the root, an unexpanded leaf
        final: set[str] = set()  # leaves decided to stay codewords
        while len(leaves) < base_size:
            open_leaves = [leaf for leaf in leaves if leaf not in final]
            if not open_leaves:
                break
            leaf = rng.choice(open_leaves)
            must_expand = len(leaf) < dmin  # too short to be a codeword
            can_expand = len(leaf) < dmax
            if not can_expand:
                final.add(leaf)
                continue
            if must_expand or rng.random() < 0.5:
                leaves.remove(leaf)
                leaves.extend([leaf + "0", leaf + "1"])
            else:
                final.add(leaf)
        if len(leaves) != base_size:
            continue
        if any(not dmin <= len(leaf) <= dmax for leaf in leaves):
            continue
        rng.shuffle(leaves)
        code = Code(tuple(leaves))
        if is_prefix_free(code):  # correct by construction; certified anyway
            return code
    raise CodeGenerationError(
        f"failed to build a prefix-free code of size {base_size} in depth range "
        f"{depth_range} within {max_tries} tries"
    )


def make_suffix_free(
    base_size: int,
    depth_range: tuple[int, int],
    rng: random.Random,
    max_tries: int = 10_000,
) -> Code:
    """Suffix-free code: build prefix-free, then reverse every codeword.

    A code that stays prefix-free after reversal is rejected and retried, so
    that the suffix-free type is genuinely distinct from the prefix-free one.
    """
    for _ in range(max_tries):
        base = make_prefix_free(base_size, depth_range, rng, max_tries=max_tries)
        code = base.reversed_code()
        if is_suffix_free(code) and not is_prefix_free(code):
            return code
    raise CodeGenerationError(
        f"failed to build a suffix-free (not prefix-free) code of size {base_size} "
        f"in depth range {depth_range} within {max_tries} tries"
    )


def make_ud(
    base_size: int,
    len_range: tuple[int, int],
    rng: random.Random,
    max_tries: int = 10_000,
) -> Code:
    """UD code that is neither prefix-free nor suffix-free, by rejection sampling.

    Random words with lengths in ``len_range`` are drawn and accepted when
    Sardinas-Patterson passes while both the prefix-free and the suffix-free
    checks fail. After ``max_tries`` rejections the known code ``{0, 01, 110}``
    is returned as a fallback (only possible when ``base_size == 3`` and the
    length range allows it).
    """
    lmin, lmax = len_range
    for _ in range(max_tries):
        words: list[str] = []
        seen: set[str] = set()
        for _ in range(base_size):
            length = rng.randint(lmin, lmax)
            word = "".join(rng.choice("01") for _ in range(length))
            if word in seen:
                break
            seen.add(word)
            words.append(word)
        if len(words) != base_size:
            continue
        code = Code(tuple(words))
        if sardinas_patterson(code) and not is_prefix_free(code) and not is_suffix_free(code):
            return code
    fallback = Code(UD_FALLBACK)
    if len(fallback) != base_size or fallback.min_len < lmin or fallback.max_len > lmax:
        raise CodeGenerationError(
            f"rejection sampling failed after {max_tries} tries and the fallback code "
            f"{UD_FALLBACK} does not match base_size={base_size} / len_range={len_range}"
        )
    return fallback


def make_code(code_cfg: CodeConfig, rng: random.Random) -> Code:
    """Dispatch to the generator for ``code_cfg.type``."""
    rng_range = code_cfg.length_range
    if code_cfg.type == "prefix-free":
        return make_prefix_free(code_cfg.base_size, rng_range, rng, code_cfg.max_tries)
    if code_cfg.type == "suffix-free":
        return make_suffix_free(code_cfg.base_size, rng_range, rng, code_cfg.max_tries)
    if code_cfg.type == "ud":
        return make_ud(code_cfg.base_size, rng_range, rng, code_cfg.max_tries)
    raise ValueError(f"unknown code type {code_cfg.type!r}")
