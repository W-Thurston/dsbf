# dsbf/eda/tasks/missingness_heatmap.py

import os
from typing import Any

import matplotlib.pyplot as plt

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


class MissingnessHeatmap(BaseTask):
    """
    Generates and saves a missingness heatmap using missingno.

    Converts Polars to Pandas if needed. Saves output image to disk.
    """

    def __init__(self, output_dir: str = "dsbf/outputs"):
        super().__init__()
        self.output_dir = output_dir

    def run(self) -> None:
        try:
            import missingno as msno

            df: Any = self.input_data

            # Convert to pandas if needed
            if is_polars(df):
                df = df.to_pandas()

            # Ensure the output directory exists
            fig_dir = os.path.join(self.output_dir, "figs")
            os.makedirs(fig_dir, exist_ok=True)

            fig_path = os.path.join(fig_dir, "missingness_heatmap.png")

            # Generate and save the heatmap
            msno.heatmap(df)
            plt.savefig(fig_path, bbox_inches="tight")
            plt.close()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary="Saved missingness heatmap to disk.",
                data={"image_path": fig_path},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Missingness heatmap failed: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
