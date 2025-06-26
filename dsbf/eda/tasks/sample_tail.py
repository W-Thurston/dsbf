# dsbf/eda/tasks/sample_tail.py

from typing import Any

from dsbf.eda.task_result import TaskResult


def sample_tail(df: Any, n: int = 5) -> TaskResult:
    """
    Returns the last n rows of the dataset as a sample.

    Args:
        df (DataFrame): Input Polars or Pandas DataFrame.
        n (int): Number of rows to return.

    Returns:
        TaskResult: Sampled tail of the dataset.
    """
    try:
        df_tail = df.tail(n)
        result = (
            df_tail.to_dict(as_series=False)
            if hasattr(df_tail, "to_dict")
            else df_tail.rows()
        )

        return TaskResult(
            name="sample_tail",
            status="success",
            summary=f"Returned last {n} rows.",
            data={"sample": result},
            metadata={"n": n},
        )

    except Exception as e:
        return TaskResult(
            name="sample_tail",
            status="failed",
            summary="Unable to compute sample_tail",
            data=None,
            metadata={"exception": type(e).__name__},
        )
