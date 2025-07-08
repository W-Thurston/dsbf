# dsbf/eda/tasks/missingness_matrix.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Missingness Matrix",
    description="Creates a matrix showing co-occurrence of missing values.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="report",
    domain="core",
    runtime_estimate="fast",
    tags=["missing", "structure", "viz"],
    expected_semantic_types=["any"],
)
class MissingnessMatrix(BaseTask):
    """
    Generates and saves a missingness matrix plot using missingno.

    Converts Polars to Pandas if needed. Saves image to disk.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(matched_col)} column(s)", "debug")

            if is_polars(df):
                df = df.to_pandas()

            fig_path = self.get_output_path("missingness_matrix.png")
            plot_result = PlotFactory.plot_missingness_matrix(
                df,
                save_path=fig_path,
                title="Missingness Matrix",
                annotations=[f"Total missing cells: {df.isnull().sum().sum()}"],
            )

            plots = {
                "missingness_matrix": {
                    "static": plot_result["path"],
                    "interactive": plot_result["plot_data"],
                }
            }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": ("Saved missingness matrix plot to disk.")},
                data={"image_path": fig_path},
                plots=plots,
                metadata={
                    "suggested_viz_type": "matrix",
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
