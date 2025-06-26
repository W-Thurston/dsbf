# dsbf/eda/tasks/summarize_numeric.py

from typing import Any

import numpy as np

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def summarize_numeric(df: Any) -> TaskResult:
    """
    Summarizes numeric columns using descriptive statistics, including extended
    percentiles and flags for near-zero variance.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.

    Returns:
        TaskResult: Dictionary of extended summary statistics per numeric column.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        numeric_df = df.select_dtypes(include=np.number)
        extended_stats = {}

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            desc = series.describe(percentiles=[0.01, 0.05, 0.95, 0.99])
            variance = np.var(series)
            near_zero_var = variance < 1e-4
            extended_stats[col] = {
                "count": desc["count"],
                "mean": desc["mean"],
                "std": desc["std"],
                "min": desc["min"],
                "1%": desc["1%"],
                "5%": desc["5%"],
                "25%": desc["25%"],
                "50%": desc["50%"],
                "75%": desc["75%"],
                "95%": desc["95%"],
                "99%": desc["99%"],
                "max": desc["max"],
                "near_zero_variance": near_zero_var,
            }

        return TaskResult(
            name="summarize_numeric",
            status="success",
            summary=f"Extended summary for {len(extended_stats)} numeric columns.",
            data=extended_stats,
        )

    except Exception as e:
        return TaskResult(
            name="summarize_numeric",
            status="failed",
            summary=f"Error during numeric summarization: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
