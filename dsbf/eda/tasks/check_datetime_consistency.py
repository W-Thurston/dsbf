# dsbf/eda/tasks/check_datetime_consistency.py

from typing import Any, Dict

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def check_datetime_consistency(df: Any) -> TaskResult:
    """
    Checks datetime columns for parseability and monotonicity.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.

    Returns:
        TaskResult: Flags and summaries per datetime column.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        results: Dict[str, Dict[str, Any]] = {}

        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                is_monotonic = (
                    parsed.is_monotonic_increasing or parsed.is_monotonic_decreasing
                )
                null_pct = parsed.isnull().mean()
                results[col] = {
                    "parseable": parsed.notnull().mean() > 0.95,
                    "monotonic": is_monotonic,
                    "null_pct_after_parse": null_pct,
                }
            except Exception:
                continue

        return TaskResult(
            name="check_datetime_consistency",
            status="success",
            summary=f"Checked datetime consistency for {len(results)} columns.",
            data=results,
        )

    except Exception as e:
        return TaskResult(
            name="check_datetime_consistency",
            status="failed",
            summary=f"Error during datetime consistency check: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
