# dsbf/eda/tasks/detect_skewness.py

from typing import Any

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def detect_skewness(df: Any) -> TaskResult:
    """
    Computes the skewness for all numeric columns.

    Args:
        df (DataFrame): Input Polars or Pandas DataFrame.

    Returns:
        TaskResult: Dictionary of columns and their skewness values.
    """
    try:
        skewness = {}

        if is_polars(df):
            import numpy as np

            for col in df.columns:
                if df[col].dtype in ("i64", "f64"):
                    series = df[col].to_numpy()
                    mean = np.mean(series)
                    std = np.std(series)
                    if std != 0:
                        skew = np.mean(((series - mean) / std) ** 3)
                        skewness[col] = skew
        else:
            from scipy.stats import skew

            numeric = df.select_dtypes(include="number")
            skewness = {col: skew(numeric[col].dropna()) for col in numeric.columns}

        return TaskResult(
            name="detect_skewness",
            status="success",
            summary=f"Computed skewness for {len(skewness)} columns.",
            data=skewness,
        )

    except Exception as e:
        return TaskResult(
            name="detect_skewness",
            status="failed",
            summary=f"Error during skewness detection: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
