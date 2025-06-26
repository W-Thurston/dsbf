# dsbf/eda/tasks/detect_high_cardinality.py

from typing import Any

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def detect_high_cardinality(df: Any, threshold: int = 50) -> TaskResult:
    """
    Detects columns with a number of unique values greater than the specified threshold.

    Args:
        df (DataFrame): Input Polars or Pandas DataFrame.
        threshold (int): Maximum number of unique values before considered high
            cardinality.

    Returns:
        TaskResult: Dictionary of high-cardinality columns and their unique counts.
    """
    results = {}
    try:
        if is_polars(df):
            for col in df.columns:
                try:
                    n_unique = df[col].n_unique()
                    if n_unique > threshold:
                        results[col] = n_unique
                except Exception:
                    continue
        else:
            for col in df.columns:
                try:
                    n_unique = df[col].nunique()
                    if n_unique > threshold:
                        results[col] = n_unique
                except Exception:
                    continue

        return TaskResult(
            name="detect_high_cardinality",
            status="success",
            summary=f"Detected {len(results)} high-cardinality columns.",
            data=results,
            metadata={"threshold": threshold},
        )

    except Exception as e:
        return TaskResult(
            name="detect_high_cardinality",
            status="failed",
            summary=f"Error during high-cardinality detection: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
