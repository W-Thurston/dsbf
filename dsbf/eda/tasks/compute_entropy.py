# dsbf/eda/tasks/compute_entropy.py

from typing import Any

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def compute_entropy(df: Any) -> TaskResult:
    """
    Computes the entropy of string-based columns to quantify categorical disorder.

    Args:
        df (DataFrame): Input Polars or Pandas DataFrame.

    Returns:
        TaskResult: Entropy values per column.
    """
    try:
        results = {}

        if is_polars(df):
            from math import log2

            for col in df.columns:
                if df[col].dtype == "str":
                    counts = df[col].value_counts()
                    total = counts["counts"].sum()
                    probs = [count / total for count in counts["counts"]]
                    entropy_val = -sum(p * log2(p) for p in probs if p > 0)
                    results[col] = entropy_val
        else:
            from scipy.stats import entropy

            for col in df.select_dtypes(include="object").columns:
                counts = df[col].value_counts()
                results[col] = entropy(counts, base=2)

        return TaskResult(
            name="compute_entropy",
            status="success",
            summary=f"Computed entropy for {len(results)} columns.",
            data=results,
        )

    except Exception as e:
        return TaskResult(
            name="compute_entropy",
            status="failed",
            summary=f"Error during entropy computation: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
