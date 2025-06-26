# dsbf/eda/tasks/detect_zeros.py

from typing import Any, Dict

import numpy as np

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def detect_zeros(df: Any, flag_threshold: float = 0.1) -> TaskResult:
    """
    Detects columns with a large number of zero values.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.
        flag_threshold (float): Proportion of zeros above which a column is flagged.

    Returns:
        TaskResult: Zero counts, flags, and percentages per numeric column.
    """
    try:
        if not hasattr(df, "shape"):
            raise ValueError("Input is not a valid dataframe.")

        n_rows = df.shape[0]
        zero_counts: Dict[str, int] = {}
        zero_percentages: Dict[str, float] = {}
        zero_flags: Dict[str, bool] = {}

        if is_polars(df):
            df = df.to_pandas()

        numeric_df = df.select_dtypes(include=[np.number])

        for col in numeric_df.columns:
            count = (numeric_df[col] == 0).sum()
            pct = count / n_rows
            zero_counts[col] = int(count)
            zero_percentages[col] = pct
            zero_flags[col] = pct > flag_threshold

        return TaskResult(
            name="detect_zeros",
            status="success",
            summary=(
                f"Flagged {sum(zero_flags.values())} " f"columns with high zero counts."
            ),
            data={
                "zero_counts": zero_counts,
                "zero_percentages": zero_percentages,
                "zero_flags": zero_flags,
            },
            metadata={
                "threshold_pct": flag_threshold,
                "total_rows": n_rows,
            },
        )

    except Exception as e:
        return TaskResult(
            name="detect_zeros",
            status="failed",
            summary=f"Error during zero detection: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
