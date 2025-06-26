# dsbf/eda/tasks/summarize_nulls.py

from typing import Any

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def summarize_nulls(df: Any) -> TaskResult:
    """
    Summarizes missing values for each column, including counts, percentages,
    high-null flags, and null pattern grouping.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.

    Returns:
        TaskResult: Summary of nulls and pattern frequencies.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        n_rows = df.shape[0]
        null_counts = df.isnull().sum().to_dict()
        null_percentages = {col: null_counts[col] / n_rows for col in df.columns}
        high_null_columns = [col for col, pct in null_percentages.items() if pct > 0.5]

        # Null pattern grouping (as a binary mask string)
        null_mask_df = df.isnull().astype(int)
        null_patterns = null_mask_df.apply(lambda row: "".join(row.astype(str)), axis=1)
        pattern_counts = null_patterns.value_counts().to_dict()

        return TaskResult(
            name="summarize_nulls",
            status="success",
            summary=f"{len(high_null_columns)} column(s) have >50% missing values.",
            data={
                "null_counts": null_counts,
                "null_percentages": null_percentages,
                "high_null_columns": high_null_columns,
                "null_patterns": pattern_counts,
            },
            metadata={"rows": n_rows},
        )

    except Exception as e:
        return TaskResult(
            name="summarize_nulls",
            status="failed",
            summary=f"Error during null summarization: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
