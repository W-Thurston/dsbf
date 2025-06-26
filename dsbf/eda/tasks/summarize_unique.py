# dsbf/eda/tasks/summarize_unique.py

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def summarize_unique(df) -> TaskResult:
    """
    Returns the number of unique values per column.

    Args:
        df (pd.DataFrame or pl.DataFrame): Input dataset.

    Returns:
        TaskResult: Unique counts per column.
    """
    try:
        if is_polars(df):
            result = {col: df[col].n_unique() for col in df.columns}
        else:
            result = df.nunique().to_dict()

        return TaskResult(
            name="summarize_unique",
            status="success",
            summary=f"Computed unique counts for {len(result)} columns.",
            data=result,
        )
    except Exception as e:
        return TaskResult(
            name="summarize_unique",
            status="failed",
            summary=str(e),
            data=None,
            metadata={"exception": type(e).__name__},
        )
