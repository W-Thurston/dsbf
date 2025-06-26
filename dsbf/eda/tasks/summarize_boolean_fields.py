# dsbf/eda/tasks/summarize_boolean_fields.py

from typing import Any, Dict

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def summarize_boolean_fields(df: Any) -> TaskResult:
    """
    Summarizes boolean columns by computing proportions of True, False,
        and missing values.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.

    Returns:
        TaskResult: Dictionary with boolean proportions per column.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        bool_cols = df.select_dtypes(include="bool").columns.tolist()
        result: Dict[str, Dict[str, float]] = {}

        for col in bool_cols:
            total = df[col].shape[0]
            true_count = df[col].sum()
            false_count = (~df[col]).sum()
            null_count = df[col].isnull().sum()

            result[col] = {
                "pct_true": true_count / total,
                "pct_false": false_count / total,
                "pct_null": null_count / total,
            }

        return TaskResult(
            name="summarize_boolean_fields",
            status="success",
            summary=f"Summarized {len(result)} boolean columns.",
            data=result,
            metadata={"bool_columns": bool_cols},
        )

    except Exception as e:
        return TaskResult(
            name="summarize_boolean_fields",
            status="failed",
            summary=f"Error during boolean field summarization: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
