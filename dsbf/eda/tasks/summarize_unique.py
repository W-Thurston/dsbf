# dsbf/eda/tasks/summarize_unique.py

from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task()
class SummarizeUnique(BaseTask):
    """
    Computes the number of unique values for each column in the DataFrame.

    Supports both Pandas and Polars input.
    """

    def run(self) -> None:
        try:
            df: Any = self.input_data

            if is_polars(df):
                result: Dict[str, int] = {col: df[col].n_unique() for col in df.columns}
            else:
                result = df.nunique().to_dict()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Computed unique counts for {len(result)} columns.",
                data=result,
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=str(e),
                data=None,
                metadata={"exception": type(e).__name__},
            )
