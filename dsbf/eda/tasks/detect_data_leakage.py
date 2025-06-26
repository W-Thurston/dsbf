# dsbf/eda/tasks/detect_data_leakage.py

from typing import Any

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def detect_data_leakage(df: Any, correlation_threshold: float = 0.99) -> TaskResult:
    """
    Detects potential data leakage by checking for highly correlated features.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.
        correlation_threshold (float): Absolute correlation above which columns are
            flagged.

    Returns:
        TaskResult: Dictionary of highly correlated column pairs.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        numeric_df = df.select_dtypes(include="number")
        corr_matrix = numeric_df.corr().abs()
        leakage_pairs = {}

        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if j > i and corr_matrix.iloc[i, j] >= correlation_threshold:
                    key = f"{col1}|{col2}"
                    leakage_pairs[key] = corr_matrix.iloc[i, j]

        return TaskResult(
            name="detect_data_leakage",
            status="success",
            summary=f"Found {len(leakage_pairs)} highly correlated feature pairs.",
            data={"leakage_pairs": leakage_pairs},
            metadata={"correlation_threshold": correlation_threshold},
        )

    except Exception as e:
        return TaskResult(
            name="detect_data_leakage",
            status="failed",
            summary=f"Error during data leakage detection: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
