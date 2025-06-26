# dsbf/eda/tasks/detect_constant_columns.py

from typing import List

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


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
                summary=f"Found {len(constant_columns)} constant column(s).",
                data={"constant_columns": constant_columns},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during constant column detection: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
