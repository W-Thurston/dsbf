# dsbf/eda/tasks/detect_constant_columns.py

from typing import List

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Constant Columns",
    description="Flags columns with a single unique value.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    tags=["redundancy", "null-equivalent"],
)
class DetectConstantColumns(BaseTask):
    """
    Identifies columns with only one unique value in the dataset.

    Works for both Polars and Pandas DataFrames.
    """

    def run(self) -> None:
        """
        Executes the constant column detection logic.
        Produces a TaskResult with a list of constant column names.
        """
        try:

            # ctx = self.context
            df = self.input_data
            constant_columns: List[str]

            if is_polars(df):
                # Use Polars' n_unique per column
                constant_columns = [
                    col for col in df.columns if df[col].n_unique() == 1
                ]
            else:
                # Pandas variant
                constant_columns = [col for col in df.columns if df[col].nunique() == 1]

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (f"Found {len(constant_columns)} constant column(s).")
                },
                data={"constant_columns": constant_columns},
                metadata={"engine": "polars" if is_polars(df) else "pandas"},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary={"message": (f"Error during constant column detection: {e}")},
                data=None,
                metadata={"exception": type(e).__name__},
            )
