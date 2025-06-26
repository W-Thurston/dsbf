# dsbf/eda/tasks/detect_duplicate_columns.py

from typing import Any, List, Tuple

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def detect_duplicate_columns(df: Any) -> TaskResult:
    """
    Detects columns that are exact duplicates of one another.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.

    Returns:
        TaskResult: List of duplicate column pairs.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        duplicate_pairs: List[Tuple[str, str]] = []
        columns = df.columns.tolist()
        seen = set()

        for i, col1 in enumerate(columns):
            for j in range(i + 1, len(columns)):
                col2 = columns[j]
                if (col1, col2) not in seen and df[col1].equals(df[col2]):
                    duplicate_pairs.append((col1, col2))
                    seen.add((col1, col2))

        return TaskResult(
            name="detect_duplicate_columns",
            status="success",
            summary=f"Found {len(duplicate_pairs)} duplicate column pair(s).",
            data={"duplicate_column_pairs": duplicate_pairs},
        )

    except Exception as e:
        return TaskResult(
            name="detect_duplicate_columns",
            status="failed",
            summary=f"Error during duplicate column detection: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
