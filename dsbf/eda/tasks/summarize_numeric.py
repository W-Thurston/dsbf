# dsbf/eda/tasks/summarize_numeric.py

from typing import Any, Dict

import numpy as np

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Summarize Numeric Columns",
    description="Computes basic stats (mean, std, min, max, etc.) for numeric columns",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["numeric", "summary"],
    expected_semantic_types=["continuous"],
)
class SummarizeNumeric(BaseTask):
    """
    Produces extended summary statistics for all numeric columns.
    Statistics include:
    - Count, mean, std, min, max
    - Percentiles: 1%, 5%, 25%, 50%, 75%, 95%, 99%
    - A flag for near-zero variance columns (variance < 1e-4)
    """

    def run(self) -> None:
        try:
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_col)} 'continuous' column(s)", "debug"
            )

            if is_polars(df):
                df = df.to_pandas()
                self._log(
                    "    Converting Polars to Pandas for numeric summarization", "debug"
                )

            numeric_df = df.select_dtypes(include=np.number)
            extended_stats: Dict[str, Dict[str, Any]] = {}
            plots: Dict[str, Dict[str, Any]] = {}

            for col in numeric_df.columns:
                series = numeric_df[col].dropna()

                if series.empty:
                    self._log(f"    {col} skipped: empty after dropna()", "debug")
                    continue

                # Compute descriptive stats with extended percentiles
                desc = series.describe(
                    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
                )
                # Custom variance check for near-constant features
                variance = np.var(series)
                near_zero_var = bool(variance < 1e-4)

                extended_stats[col] = {
                    "count": desc.get("count", np.nan),
                    "mean": desc.get("mean", np.nan),
                    "std": desc.get("std", np.nan),
                    "min": desc.get("min", np.nan),
                    "1%": desc.get("1%", np.nan),
                    "5%": desc.get("5%", np.nan),
                    "25%": desc.get("25%", np.nan),
                    "50%": desc.get("50%", np.nan),
                    "75%": desc.get("75%", np.nan),
                    "95%": desc.get("95%", np.nan),
                    "99%": desc.get("99%", np.nan),
                    "max": desc.get("max", np.nan),
                    "near_zero_variance": near_zero_var,
                }

                # Add visualizations
                save_path = self.get_output_path(f"{col}_histogram.png")
                static = PlotFactory.plot_histogram_static(series, save_path)["path"]
                interactive = PlotFactory.plot_histogram_interactive(series)
                plots[col] = {
                    "static": static,
                    "interactive": interactive,
                }

            self._log(f"    Summarized {len(extended_stats)} numeric columns", "debug")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Extended summary for {len(extended_stats)} numeric columns."
                    )
                },
                data=extended_stats,
                plots=plots,
                metadata={
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Summary",
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
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} — {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
