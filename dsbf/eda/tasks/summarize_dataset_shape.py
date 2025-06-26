# dsbf/eda/tasks/summarize_dataset_shape.py

from typing import Any

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def summarize_dataset_shape(df: Any) -> TaskResult:
    """
    Summarizes overall dataset shape and structure characteristics.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.

    Returns:
        TaskResult: Summary of row/column count, memory usage, and basic structure
            stats.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        n_rows, n_cols = df.shape
        null_pct = df.isnull().sum().sum() / (n_rows * n_cols)
        mem_bytes = df.memory_usage(deep=True).sum()

        return TaskResult(
            name="summarize_dataset_shape",
            status="success",
            summary=f"Dataset has {n_rows} rows and {n_cols} columns.",
            data={
                "num_rows": n_rows,
                "num_columns": n_cols,
                "null_cell_percentage": round(null_pct, 4),
                "approx_memory_MB": round(mem_bytes / 1_048_576, 2),
            },
        )

    except Exception as e:
        return TaskResult(
            name="summarize_dataset_shape",
            status="failed",
            summary=f"Error during dataset shape summarization: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
