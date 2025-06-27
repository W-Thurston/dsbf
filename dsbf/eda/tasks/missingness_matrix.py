# dsbf/eda/tasks/missingness_matrix.py

from typing import Any

import matplotlib.pyplot as plt

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Missingness Matrix",
    description="Creates a matrix showing co-occurrence of missing values.",
    depends_on=["infer_types"],
    stage="report",
    tags=["missing", "structure", "viz"],
)
class MissingnessMatrix(BaseTask):
    """
    Generates and saves a missingness matrix plot using missingno.

    Converts Polars to Pandas if needed. Saves image to disk.
    """

    def run(self) -> None:
        try:
            import missingno as msno

            df: Any = self.input_data

            if is_polars(df):
                df = df.to_pandas()

            fig_path = self.get_output_path("missingness_matrix.png")

            msno.matrix(df)
            plt.savefig(fig_path, bbox_inches="tight")
            plt.close()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary="Saved missingness matrix plot to disk.",
                data={"image_path": fig_path},
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Missingness matrix failed: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
