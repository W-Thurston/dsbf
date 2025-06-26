# dsbf/eda/tasks/summarize_modes.py

from typing import Any

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def summarize_modes(df: Any) -> TaskResult:
    """
    Summarizes the mode (most frequent value) for each column.

    Args:
        df (DataFrame): Input Polars or Pandas DataFrame.

    Returns:
        TaskResult: Dictionary of column modes.
    """
    try:
        if is_polars(df):
            result = {col: df[col].mode().to_list() for col in df.columns}
        else:
            result = df.mode().iloc[0].to_dict()

        return TaskResult(
            name="summarize_modes",
            status="success",
            summary=f"Computed mode(s) for {len(result)} columns.",
            data=result,
        )

    except Exception as e:
        return TaskResult(
            name="summarize_modes",
            status="failed",
            summary=f"Error during mode summarization: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
