# dsbf/eda/tasks/sample_head.py

from typing import Any

from dsbf.eda.task_result import TaskResult


def sample_head(df: Any, n: int = 5) -> TaskResult:
    """
    Returns the first n rows of the dataset as a sample.

    Args:
        df (DataFrame): Input Polars or Pandas DataFrame.
        n (int): Number of rows to return.

    Returns:
        TaskResult: Sampled head of the dataset.
    """
    try:
        df_head = df.head(n)
        result = (
            df_head.to_dict(as_series=False)
            if hasattr(df_head, "to_dict")
            else df_head.rows()
        )

        return TaskResult(
            name="sample_head",
            status="success",
            summary=f"Returned first {n} rows.",
            data={"sample": result},
            metadata={"n": n},
        )

    except Exception as e:
        return TaskResult(
            name="sample_head",
            status="failed",
            summary="Unable to compute sample_head",
            data=None,
            metadata={"exception": type(e).__name__},
        )
