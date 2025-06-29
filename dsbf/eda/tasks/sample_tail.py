# dsbf/eda/tasks/sample_tail.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Sample Tail",
    description="Returns the last N rows of the dataset.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    tags=["preview"],
)
class SampleTail(BaseTask):
    """
    Returns the last N rows of the dataset for preview.
    Works with both Pandas and Polars.
    """

    def run(self) -> None:
        try:
            # ctx = self.context
            df = self.input_data

            n = self.get_task_param("n")
            if n is None:
                n = 5
            else:
                n = int(n)

            if n == 0:
                df_tail = df.head(0)
            else:
                df_tail = df.tail(n)

            if is_polars(df_tail):
                result = df_tail.to_pandas().to_dict(orient="list")
            else:
                result = df_tail.to_dict(orient="list")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": (f"Returned last {n} rows.")},
                data={"sample": result},
                metadata={"n": n},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary={"message": ("Unable to compute sample_tail")},
                data=None,
                metadata={"exception": type(e).__name__},
            )
