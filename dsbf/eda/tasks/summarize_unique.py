# dsbf/eda/tasks/summarize_unique.py

from typing import Any, Dict

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Summarize Unique Values",
    description="Reports unique value counts per column.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["uniqueness", "summary"],
    expected_semantic_types=["any"],
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

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(matched_col)} column(s)", "debug")

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
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Summary",
                    "display_priority": "low",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
            )

            # Use the output data dict to build a plot
            if self.context and self.context.output_dir and self.output.data:
                counts_series = pd.Series(self.output.data)

                # Add annotations based on cardinality
                annotations = []
                for col, count in self.output.data.items():
                    if count == 1:
                        annotations.append(f"{col} is constant (1 unique value)")
                    elif count > 50:
                        annotations.append(
                            f"{col} has high cardinality ({count} values)"
                        )

                save_path = self.get_output_path("unique_counts_barplot.png")
                static = PlotFactory.plot_barplot_static(counts_series, save_path)
                interactive = PlotFactory.plot_barplot_interactive(
                    counts_series, annotations=annotations
                )

                self.output.plots = {
                    "unique_counts": {
                        "static": static["path"],
                        "interactive": interactive,
                    }
                }

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
