# dsbf/eda/tasks/detect_duplicates.py

from typing import Any

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def detect_duplicates(df: Any) -> TaskResult:
    """
    Counts the number of duplicate rows in the dataset.

    Args:
        df (DataFrame): Input Polars or Pandas DataFrame.

    Returns:
        TaskResult: Total number of duplicate rows.
    """
    try:
        if is_polars(df):
            duplicates = df.shape[0] - df.unique().shape[0]
        else:
            duplicates = df.duplicated().sum()

        return TaskResult(
            name="detect_duplicates",
            status="success",
            summary=f"Found {duplicates} duplicate rows.",
            data={"duplicate_count": duplicates},
        )

    except Exception as e:
        return TaskResult(
            name="detect_duplicates",
            status="failed",
            summary=f"Error during duplicate detection: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
