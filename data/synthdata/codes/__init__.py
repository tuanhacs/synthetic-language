"""Codes: model, generators (one per code type) and certifiers."""

from .certify import (
    CodeReport,
    certify,
    is_prefix_free,
    is_suffix_free,
    sardinas_patterson,
)
from .generate import (
    CodeGenerationError,
    make_code,
    make_prefix_free,
    make_suffix_free,
    make_ud,
)
from .model import Code, load_codeword_pool

__all__ = [
    "Code",
    "CodeReport",
    "CodeGenerationError",
    "certify",
    "is_prefix_free",
    "is_suffix_free",
    "load_codeword_pool",
    "sardinas_patterson",
    "make_code",
    "make_prefix_free",
    "make_suffix_free",
    "make_ud",
]
