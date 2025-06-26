# dsbf/eda/tasks/detect_out_of_bounds.py

from typing import Any, Dict, Optional, Tuple

import numpy as np

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars

DEFAULT_RULES: Dict[str, Tuple[float, float]] = {
    "age": (0, 120),
    "temperature": (-100, 150),
    "percent": (0, 100),
    "score": (0, 1),
}


def detect_out_of_bounds(
    df: Any, custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> TaskResult:
    """
    Detects numeric columns with values outside expected bounds.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.
        custom_bounds (dict): Optional dictionary of column-specific bounds.

    Returns:
        TaskResult: Dictionary of out-of-bounds values per column.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        bounds = {**DEFAULT_RULES, **(custom_bounds or {})}
        flagged: Dict[str, Dict[str, Any]] = {}

        for col in df.select_dtypes(include=np.number).columns:
            if col in bounds:
                lower, upper = bounds[col]
                series = df[col].dropna()
                violations = series[(series < lower) | (series > upper)]
                if not violations.empty:
                    flagged[col] = {
                        "count": int(violations.count()),
                        "min_violation": float(violations.min()),
                        "max_violation": float(violations.max()),
                        "expected_range": (float(lower), float(upper)),
                    }

        return TaskResult(
            name="detect_out_of_bounds",
            status="success",
            summary=f"Detected {len(flagged)} column(s) with out-of-bounds values.",
            data=flagged,
            metadata={"rule_columns": list(bounds.keys())},
        )

    except Exception as e:
        return TaskResult(
            name="detect_out_of_bounds",
            status="failed",
            summary=f"Error during bounds checking: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
