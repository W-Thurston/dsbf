# dsbf/eda/tasks/summarize_value_counts.py

from typing import Any, Dict

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def summarize_value_counts(df: Any, top_k: int = 5) -> TaskResult:
    """
    Returns the top-k most frequent values per column.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.
        top_k (int): Number of top values to return per column.

    Returns:
        TaskResult: Dictionary of value counts per column.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        result: Dict[str, Dict[str, int]] = {}

        for col in df.columns:
            try:
                vc = df[col].value_counts(dropna=False).head(top_k)
                result[col] = vc.to_dict()
            except Exception:
                continue

        return TaskResult(
            name="summarize_value_counts",
            status="success",
            summary=f"Computed value counts for {len(result)} columns.",
            data=result,
            metadata={"top_k": top_k},
        )

    except Exception as e:
        return TaskResult(
            name="summarize_value_counts",
            status="failed",
            summary=f"Error during value count summarization: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
