# dsbf/eda/tasks/summarize_nulls.py

from typing import Any, Dict, List

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Summarize Nulls",
    description="Reports null value counts per column.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["nulls", "missing"],
    expected_semantic_types=["any"],
)
class SummarizeNulls(BaseTask):
    """
    Identifies and summarizes missing values in a dataset.

    Computes:
    - Null counts per column
    - Null percentages per column
    - Columns with >50% missing values
    - Row-wise null patterns as binary strings (e.g., '101')
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(matched_col)} column(s)", "debug")

            null_threshold = float(self.get_task_param("null_threshold") or 0.5)

            if is_polars(df):
                df = df.to_pandas()

            n_rows: int = df.shape[0]

            # Column null counts and percentages
            null_counts: Dict[str, int] = df.isnull().sum().to_dict()
            null_percentages: Dict[str, float] = {
                col: null_counts[col] / n_rows for col in df.columns
            }

            high_null_columns: List[str] = [
                col for col, pct in null_percentages.items() if pct >= null_threshold
            ]
            self._log(
                f"Detected {len(high_null_columns)} columns with >50% nulls", "debug"
            )

            # Row-wise null pattern frequency (e.g., "101" means null in cols 1 and 3)
            null_mask_df = df.isnull().astype(int)
            null_patterns = null_mask_df.apply(
                lambda row: "".join(row.astype(str)), axis=1
            )
            pattern_counts: Dict[str, int] = null_patterns.value_counts().to_dict()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"{len(high_null_columns)} column(s) have >50% missing values."
                    )
                },
                data={
                    "null_counts": null_counts,
                    "null_percentages": null_percentages,
                    "high_null_columns": high_null_columns,
                    "null_patterns": pattern_counts,
                },
                metadata={
                    "rows": n_rows,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Missingness",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
            )

            if self.context and self.context.output_dir and self.output.data:
                null_series = pd.Series(self.output.data["null_counts"])
                save_path = self.get_output_path("null_counts_barplot.png")

                # Annotate fully null or high-null columns
                annotations = []
                for col, count in self.output.data["null_counts"].items():
                    pct = self.output.data["null_percentages"].get(col, 0)
                    if count == self.output.metadata["rows"]:
                        annotations.append(f"{col} is fully null ({pct:.1%})")
                    elif pct > 0.5:
                        annotations.append(f"{col} has >50% missing ({pct:.1%})")

                # Static and interactive plots
                static = PlotFactory.plot_barplot_static(null_series, save_path)
                interactive = PlotFactory.plot_barplot_interactive(
                    null_series,
                    annotations=annotations,
                )

                self.output.plots = {
                    "null_counts": {
                        "static": static["path"],
                        "interactive": interactive,
                    }
                }

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
