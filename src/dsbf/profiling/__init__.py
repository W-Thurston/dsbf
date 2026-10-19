# dsbf/profiling/__init__.py
"""The profiling pass: what DSBF learns about each column before any check runs."""

from dsbf.profiling.columns import DEFAULT_QUANTILES, profile_columns
from dsbf.profiling.dataset import (
    DEFAULT_MAX_PAIR_COLUMNS,
    DEFAULT_MAX_PAIRS,
    profile_dataset,
)
from dsbf.profiling.stage import StageThresholds, infer_stage
from dsbf.profiling.types import (
    TYPE_SAMPLE_ROWS,
    ColumnType,
    infer_column_type,
    infer_column_types,
)

__all__ = [
    "DEFAULT_MAX_PAIRS",
    "DEFAULT_MAX_PAIR_COLUMNS",
    "DEFAULT_QUANTILES",
    "TYPE_SAMPLE_ROWS",
    "ColumnType",
    "StageThresholds",
    "infer_stage",
    "infer_column_type",
    "infer_column_types",
    "profile_columns",
    "profile_dataset",
]
