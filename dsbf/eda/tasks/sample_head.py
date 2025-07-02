# dsbf/eda/tasks/sample_head.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Sample Head",
    description="Returns the first N rows of the dataset.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["preview"],
)
class SampleHead(BaseTask):
    """
    Returns the first N rows of the dataset for preview.
    Works with both Pandas and Polars.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df = self.input_data

            n = int(self.get_task_param("n") or 5)

            df_head = df.head(n)
            self._log(f"Returning first {n} rows", "debug")

            if is_polars(df_head):
                result = df_head.to_pandas().to_dict(orient="list")
            else:
                result = df_head.to_dict(orient="list")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": (f"Returned first {n} rows.")},
                data={"sample": result},
                metadata={"n": n},
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
