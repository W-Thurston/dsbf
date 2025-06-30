# dsbf/eda/tasks/summarize_unique.py

from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Summarize Unique Values",
    description="Reports unique value counts per column.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    tags=["uniqueness", "summary"],
)
class SummarizeUnique(BaseTask):
    """
    Computes the number of unique values for each column in the DataFrame.

    Supports both Pandas and Polars input.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            if is_polars(df):
                result: Dict[str, int] = {col: df[col].n_unique() for col in df.columns}
                self._log(
                    f"Computing unique values for {len(df.columns)} columns", "debug"
                )
            else:
                result = df.nunique().to_dict()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (f"Computed unique counts for {len(result)} columns.")
                },
                data=result,
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
