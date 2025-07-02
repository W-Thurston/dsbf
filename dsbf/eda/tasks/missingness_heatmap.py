# dsbf/eda/tasks/missingness_heatmap.py

from typing import Any

import matplotlib.pyplot as plt
import missingno as msno

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Missingness Heatmap",
    description="Visualizes missing values with a heatmap.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="report",
    domain="core",
    runtime_estimate="fast",
    tags=["missing", "viz"],
)
class MissingnessHeatmap(BaseTask):
    """
    Generates and saves a missingness heatmap using missingno.

    Converts Polars to Pandas if needed. Saves output image to disk.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            # Convert to pandas if needed
            if is_polars(df):
                df = df.to_pandas()

            fig_path = self.get_output_path("missingness_heatmap.png")

            # Generate and save the heatmap
            msno.heatmap(df)
            plt.savefig(fig_path, bbox_inches="tight")
            plt.close()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": ("Saved missingness heatmap to disk.")},
                data={"image_path": fig_path},
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
