# dsbf/eda/tasks/detect_duplicates.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Duplicates",
    description="Detects duplicated rows in the dataset.",
    depends_on=["infer_types"],
    stage="raw",
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
                duplicate_count = df.shape[0] - df.unique().shape[0]
            else:
                # In Pandas, use .duplicated() to count duplicate rows
                duplicate_count = df.duplicated().sum()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Found {duplicate_count} duplicate row(s).",
                data={"duplicate_count": duplicate_count},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during duplicate detection: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
