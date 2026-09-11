"""synthdata — synthetic language engine and dataset generation.

The language: sentences are walks on a hidden graph, each step encoded by one
codeword from the current vertex's codebook and concatenated without separators.
See ``../README.md`` and ``../../docs/context.md``.
"""

from .codebooks import (
    Codebooks,
    CodebookReport,
    assign,
    assign_arbitrary_overlap,
    assign_disjoint_random,
)
from .codes import Code, CodeReport, certify, load_codeword_pool, make_code
from .config import (
    CodeConfig,
    Config,
    ConfigError,
    DataConfig,
    LanguageConfig,
    OverlapConfig,
    NoiseConfig,
    TaskConfig,
    config_hash,
    load_config,
    parse_config,
)
from .dataset import Splits, build_pool, nested_subsets, pool_stats, split_pool, stream
from .graphs import GridGraph, make_grid
from .language import Language, LanguageReport, Sample
from .noise import apply_noise
from .qa import (
    CrossRegionLayout,
    QASample,
    build_qa_splits,
    cross_region_direction,
    cross_region_layout,
    cross_region_segment_count,
    is_piecewise_simple_answer,
    sample_qa,
    validate_qa_answer,
    waypoints_in_order,
)
from .storage import load_dataset, save_dataset
from .tokenizer import BitTokenizer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BitTokenizer",
    "Code",
    "CodeConfig",
    "CodeReport",
    "Codebooks",
    "CodebookReport",
    "Config",
    "ConfigError",
    "DataConfig",
    "GridGraph",
    "Language",
    "LanguageConfig",
    "OverlapConfig",
    "LanguageReport",
    "NoiseConfig",
    "TaskConfig",
    "CrossRegionLayout",
    "QASample",
    "Sample",
    "Splits",
    "apply_noise",
    "assign",
    "assign_arbitrary_overlap",
    "assign_disjoint_random",
    "build_pool",
    "build_qa_splits",
    "cross_region_direction",
    "cross_region_layout",
    "cross_region_segment_count",
    "certify",
    "config_hash",
    "load_config",
    "load_codeword_pool",
    "load_dataset",
    "make_code",
    "make_grid",
    "nested_subsets",
    "parse_config",
    "pool_stats",
    "save_dataset",
    "sample_qa",
    "validate_qa_answer",
    "waypoints_in_order",
    "is_piecewise_simple_answer",
    "split_pool",
    "stream",
]
