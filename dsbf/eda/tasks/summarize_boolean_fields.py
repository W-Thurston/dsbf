from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def is_boolean_column(series) -> bool:
    """Identify columns that only contain True/False or nulls."""
    non_null_vals = series.dropna().unique()
    return set(non_null_vals).issubset({True, False})


@register_task()
class SummarizeBooleanFields(BaseTask):
    """
    Summarizes boolean columns by computing proportions of True, False, and
    missing values.
    """

    def run(self) -> None:
        try:
            df: Any = self.input_data

            if is_polars(df):
                df = df.to_pandas()

            bool_cols = [col for col in df.columns if is_boolean_column(df[col])]
            result: Dict[str, Dict[str, float]] = {}

            for col in bool_cols:
                total = len(df[col])
                true_count = (df[col] == True).sum()  # noqa: E712
                false_count = (df[col] == False).sum()  # noqa: E712
                null_count = df[col].isnull().sum()

                result[col] = {
                    "pct_true": true_count / total,
                    "pct_false": false_count / total,
                    "pct_null": null_count / total,
                }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Summarized {len(result)} boolean columns.",
                data=result,
                metadata={"bool_columns": bool_cols},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during boolean field summarization: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
