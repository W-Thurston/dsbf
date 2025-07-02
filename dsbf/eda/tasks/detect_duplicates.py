# dsbf/eda/tasks/detect_duplicates.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Duplicates",
    description="Detects duplicated rows in the dataset.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["duplicates", "rows"],
)
class DetectDuplicates(BaseTask):
    """
    Detects and counts duplicate rows in the dataset.
    """

    def run(self) -> None:
        """
        Run the duplicate detection logic.

        Sets output to a TaskResult with the count of duplicate rows.
        """
        try:
            df: Any = self.input_data
            if is_polars(df):
                # In Polars, duplicates = total rows - unique rows
                total_rows = df.shape[0]
                unique_rows = df.unique(subset=None, maintain_order=True).shape[0]
                duplicate_count = total_rows - unique_rows
            else:
                # In Pandas, use .duplicated() to count duplicate rows
                duplicate_count = df.duplicated().sum()

            self._log(f"Duplicate count: {duplicate_count}", "debug")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Found {duplicate_count} duplicate row(s)."},
                data={"duplicate_count": duplicate_count},
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
