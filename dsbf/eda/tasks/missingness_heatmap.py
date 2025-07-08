# dsbf/eda/tasks/missingness_heatmap.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Missingness Heatmap",
    description="Visualizes missing values with a heatmap.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="report",
    domain="core",
    runtime_estimate="fast",
    tags=["missing", "viz"],
    expected_semantic_types=["any"],
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

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(matched_col)} column(s)", "debug")

            # Convert to pandas if needed
            if is_polars(df):
                df = df.to_pandas()

            missing_cells = df.isnull().sum().sum()
            annotation = [f"Total missing cells: {missing_cells}"]

            save_path = self.get_output_path("missingness_heatmap.png")

            static = PlotFactory.plot_null_matrix_static(
                df, save_path=save_path, title="Missingness Heatmap"
            )
            interactive = PlotFactory.plot_null_matrix_interactive(
                df, title="Missingness Heatmap", annotations=annotation
            )

            plots = {
                "missingness_heatmap": {
                    "static": static["path"],
                    "interactive": interactive,
                }
            }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Found {missing_cells} missing cells in dataset."},
                data={"missing_cells": int(missing_cells)},
                plots=plots,
                metadata={
                    "suggested_viz_type": "heatmap",
                    "recommended_section": "Missingness",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
