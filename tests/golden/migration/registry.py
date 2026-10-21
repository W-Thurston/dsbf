"""Which comparator checks each core check, and which differences are accepted."""

from __future__ import annotations

from tests.golden.migration.accepted import ACCEPTED
from tests.golden.migration.column_health import (
    compare_boolean_fields,
    compare_constant_columns,
    compare_duplicate_columns,
    compare_duplicate_rows,
    compare_high_cardinality,
    compare_id_columns,
    compare_near_zero_variance,
    compare_null_rate,
    compare_single_dominant_value,
)
from tests.golden.migration.distribution import (
    compare_dtype_optimizations,
    compare_entropy,
    compare_kurtosis,
    compare_not_run,
    compare_sample_size,
    compare_skewness,
    compare_summarize_numeric,
)
from tests.golden.migration.harness import Accepted, Comparator
from tests.golden.migration.zeros import compare_zeros

# Legacy task to its comparator. The core check that replaces the task is
# ``dsbf.legacy.SUPERSEDED_BY``'s value for it; two tasks can share a check, and a
# task can be replaced by several.
COMPARATORS: dict[str, Comparator] = {
    "detect_zeros": compare_zeros,
    "summarize_nulls": compare_null_rate,
    "detect_constant_columns": compare_constant_columns,
    "detect_high_cardinality": compare_high_cardinality,
    "detect_id_columns": compare_id_columns,
    "detect_single_dominant_value": compare_single_dominant_value,
    "summarize_boolean_fields": compare_boolean_fields,
    "detect_near_zero_variance": compare_near_zero_variance,
    "detect_duplicates": compare_duplicate_rows,
    "detect_duplicate_columns": compare_duplicate_columns,
    "detect_skewness": compare_skewness,
    "summarize_numeric": compare_summarize_numeric,
    "compute_kurtosis": compare_kurtosis,
    "compute_entropy": compare_entropy,
    "suggest_dtype_optimizations": compare_dtype_optimizations,
    "sample_size_adequacy": compare_sample_size,
    "detect_class_imbalance": compare_not_run,
    "schema_validation": compare_not_run,
}

__all__ = ["ACCEPTED", "COMPARATORS", "Accepted"]
