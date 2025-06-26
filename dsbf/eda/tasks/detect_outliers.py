# dsbf/eda/tasks/detect_outliers.py

from typing import Any, Dict, List

import numpy as np

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def detect_outliers(
    df: Any, method: str = "iqr", flag_threshold: float = 0.01
) -> TaskResult:
    """
    Detects outliers in numeric columns using the IQR method.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.
        method (str): Method for outlier detection. Currently supports 'iqr'.
        flag_threshold (float): Proportion of outliers above which a column is flagged.

    Returns:
        TaskResult: Outlier counts, flags, and row indices.
    """
    try:
        if not hasattr(df, "shape"):
            raise ValueError("Input is not a valid dataframe.")

        n_rows = df.shape[0]
        outlier_counts: Dict[str, int] = {}
        outlier_flags: Dict[str, bool] = {}
        outlier_rows: Dict[str, List[int]] = {}

        if is_polars(df):
            df = df.to_pandas()

        numeric_df = df.select_dtypes(include=[np.number])

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (series < lower) | (series > upper)
            indices = series[mask].index.tolist()
            outlier_rows[col] = indices
            outlier_counts[col] = len(indices)
            outlier_flags[col] = len(indices) > flag_threshold * n_rows

        return TaskResult(
            name="detect_outliers",
            status="success",
            summary=f"Detected outliers in {sum(outlier_flags.values())} column(s).",
            data={
                "outlier_counts": outlier_counts,
                "outlier_flags": outlier_flags,
                "outlier_rows": outlier_rows,
            },
            metadata={
                "method": method,
                "threshold_pct": flag_threshold,
                "total_rows": n_rows,
            },
        )

    except Exception as e:
        return TaskResult(
            name="detect_outliers",
            status="failed",
            summary=f"Error during outlier detection: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
