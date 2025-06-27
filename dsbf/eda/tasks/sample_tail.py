# dsbf/eda/tasks/sample_tail.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task()
class SampleTail(BaseTask):
    """
    Returns the last N rows of the dataset for preview.
    Works with both Pandas and Polars.
    """

    def __init__(self, n: int = 5):
        super().__init__()
        self.n = n

    def run(self) -> None:
        try:
            df: Any = self.input_data
            df_tail = df.tail(self.n)

            if is_polars(df_tail):
                result = df_tail.to_pandas().to_dict(orient="list")
            else:
                result = df_tail.to_dict(orient="list")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Returned last {self.n} rows.",
                data={"sample": result},
                metadata={"n": self.n},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary="Unable to compute sample_tail",
                data=None,
                metadata={"exception": type(e).__name__},
            )
