"""Which comparator checks each core check, and which differences are accepted."""

from __future__ import annotations

from tests.golden.migration.accepted import ACCEPTED as _ACCEPTED
from tests.golden.migration.accepted_analysis import ACCEPTED_ANALYSIS
from tests.golden.migration.accepted_associations import ACCEPTED_ASSOCIATIONS
from tests.golden.migration.analysis import (
    compare_bimodality,
    compare_normality,
    compare_outliers,
    compare_qq,
)
from tests.golden.migration.associations import (
    compare_contingency_tables,
    compare_data_leakage,
    compare_pairwise_associations,
)
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

ACCEPTED = (*_ACCEPTED, *ACCEPTED_ANALYSIS, *ACCEPTED_ASSOCIATIONS)

# Tasks whose checks cover different parts of what the task did: the task is compared
# wherever any of its checks ran.
PARTIAL_COVERAGE = frozenset({"compute_pairwise_associations"})

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
    "detect_outliers": compare_outliers,
    "normality_tests": compare_normality,
    "normality_qq_plots": compare_qq,
    "detect_bimodal_distribution": compare_bimodality,
    "detect_out_of_bounds": compare_not_run,
    "compute_pairwise_associations": compare_pairwise_associations,
    "contingency_tables": compare_contingency_tables,
    "detect_data_leakage": compare_data_leakage,
}

__all__ = ["ACCEPTED", "COMPARATORS", "PARTIAL_COVERAGE", "Accepted"]
