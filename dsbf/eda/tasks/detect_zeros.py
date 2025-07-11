# dsbf/eda/tasks/detect_zeros.py

from typing import Any, Dict

import numpy as np
import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Detect Zeros",
    description="Flags columns or rows with high zero concentration.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["zeros", "sparsity"],
    expected_semantic_types=["continuous"],
)
class DetectZeros(BaseTask):
    """
    Detects numeric columns with a high proportion of zero values.
    Flags columns where zeros exceed a specified threshold.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_col)} 'continuous' column(s)", "debug"
            )

            flag_threshold = float(self.get_task_param("flag_threshold") or 0.95)

            if is_polars(df):
                df = df.to_pandas()

            if not hasattr(df, "shape"):
                raise ValueError("Input is not a valid dataframe.")

            n_rows = df.shape[0]
            zero_counts: Dict[str, int] = {}
            zero_percentages: Dict[str, float] = {}
            zero_flags: Dict[str, bool] = {}

            numeric_df = df.select_dtypes(include=[np.number])

            for col in numeric_df.columns:
                count = int((numeric_df[col] == 0).sum())
                pct = count / n_rows
                zero_counts[col] = count
                zero_percentages[col] = pct
                zero_flags[col] = pct > flag_threshold

            plots: dict[str, dict[str, Any]] = {}

            # Create a Series for plotting: index=columns, values=% zeros
            percent_series = pd.Series(zero_percentages).sort_values(ascending=False)
            percent_series.name = "% Zeros"

            # Only plot if there are any non-zero entries
            if not percent_series.empty and percent_series.max() > 0:
                annotation = [f"Threshold: {flag_threshold:.0%}"]
                save_path = self.get_output_path("zero_percentage_barplot.png")

                static = PlotFactory.plot_barplot_static(
                    percent_series,
                    save_path=save_path,
                    title="Zero Concentration by Column",
                )
                interactive = PlotFactory.plot_barplot_interactive(
                    percent_series,
                    title="Zero Concentration by Column",
                    annotations=annotation,
                )

                plots["zero_percentages"] = {
                    "static": static["path"],
                    "interactive": interactive,
                }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Flagged {sum(zero_flags.values())}"
                        f"column(s) with high zero counts."
                    )
                },
                data={
                    "zero_counts": zero_counts,
                    "zero_percentages": zero_percentages,
                    "zero_flags": zero_flags,
                },
                metadata={
                    "threshold_pct": flag_threshold,
                    "total_rows": n_rows,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Sparsity",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
                plots=plots,
            )

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} — {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
