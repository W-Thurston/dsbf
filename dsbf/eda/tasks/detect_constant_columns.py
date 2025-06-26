# dsbf/eda/tasks/detect_constant_columns.py

from typing import Any

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def detect_constant_columns(df: Any) -> TaskResult:
    """
    Identifies columns with only one unique value.

    Args:
        df (DataFrame): Input Polars or Pandas DataFrame.

    Returns:
        TaskResult: List of constant columns.
    """
    try:
        if is_polars(df):
            result = [col for col in df.columns if df[col].n_unique() == 1]
        else:
            result = [col for col in df.columns if df[col].nunique() == 1]

        return TaskResult(
            name="detect_constant_columns",
            status="success",
            summary=f"Found {len(result)} constant columns.",
            data={"constant_columns": result},
        )

    except Exception as e:
        return TaskResult(
            name="detect_constant_columns",
            status="failed",
            summary=f"Error during constant column detection: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
