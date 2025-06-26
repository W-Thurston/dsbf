# dsbf/eda/tasks/detect_id_columns.py

from typing import Any

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def detect_id_columns(df: Any) -> TaskResult:
    """
    Detects columns likely to be unique identifiers (e.g. IDs).

    Args:
        df (DataFrame): Input Polars or Pandas DataFrame.

    Returns:
        TaskResult: Dictionary of likely ID columns and their unique counts.
    """
    results = {}
    try:
        n_rows = df.shape[0]
        if is_polars(df):
            for col in df.columns:
                try:
                    n_unique = df[col].n_unique()
                    if n_unique >= 0.95 * n_rows:
                        results[col] = f"{n_unique} unique values (likely ID)"
                except Exception:
                    continue
        else:
            for col in df.columns:
                try:
                    n_unique = df[col].nunique()
                    if n_unique >= 0.95 * n_rows:
                        results[col] = f"{n_unique} unique values (likely ID)"
                except Exception:
                    continue

        return TaskResult(
            name="detect_id_columns",
            status="success",
            summary=f"Detected {len(results)} likely ID columns.",
            data=results,
            metadata={"rows": n_rows},
        )

    except Exception as e:
        return TaskResult(
            name="detect_id_columns",
            status="failed",
            summary=f"Error during ID column detection: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
