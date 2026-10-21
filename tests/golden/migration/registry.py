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
from tests.golden.migration.harness import Accepted, Comparator
from tests.golden.migration.zeros import compare_zeros

# Legacy task to its comparator. The core check that replaces the task is
# ``dsbf.legacy.SUPERSEDED_BY``'s value for it; two tasks can share a check.
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
}

__all__ = ["ACCEPTED", "COMPARATORS", "Accepted"]
