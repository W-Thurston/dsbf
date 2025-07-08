# dsbf/eda/tasks/categorical_length_stats.py

from typing import Any, Dict

import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Categorical Length Stats",
    description="Computes string length statistics for text-like categorical columns.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["categorical", "text", "stats"],
    expected_semantic_types=["categorical", "text"],
)
class CategoricalLengthStats(BaseTask):
    """
    Computes string length statistics (mean, min, max) for all text-like categorical
    columns. Supports both Pandas and Polars DataFrames.

    Produces a TaskResult with per-column summary stats and length histograms.
    """

    def run(self) -> None:
        """
        Run the task and populate self.output with a TaskResult.
        This method filters for categorical/text-like columns based on semantic types,
        computes string length stats for each, and attaches static/interactive plots.
        """
        df = self.input_data
        results: Dict[str, Dict[str, float]] = {}
        plots: Dict[str, Dict[str, Any]] = {}

        try:
            # Select matching columns based on semantic type
            matching_cols, excluded = self.get_columns_by_intent()

            # Compute string length stats per column
            for col in matching_cols:
                try:
                    if is_polars(df):
                        lengths = df.select(
                            pl.col(col).cast(pl.Utf8).str.len_chars().alias("len")
                        ).drop_nulls()["len"]
                        if lengths.len() == 0:
                            continue
                        stats = {
                            "mean_length": lengths.mean(),
                            "max_length": lengths.max(),
                            "min_length": lengths.min(),
                        }
                    else:
                        lengths = df[col].dropna().str.len()
                        if len(lengths) == 0:
                            continue
                        stats = {
                            "mean_length": lengths.mean(),
                            "max_length": lengths.max(),
                            "min_length": lengths.min(),
                        }

                    results[col] = stats

                    # Generate histogram plots
                    lengths_series = (
                        lengths.to_pandas()
                        if hasattr(lengths, "to_pandas")
                        else lengths
                    )
                    lengths_series.name = f"{col} length"

                    annotation = [
                        f"Min: {stats['min_length']:.1f}, "
                        f"Mean: {stats['mean_length']:.1f}, "
                        f"Max: {stats['max_length']:.1f}"
                    ]

                    save_path = self.get_output_path(f"{col}_length_hist.png")
                    static = PlotFactory.plot_histogram_static(
                        lengths_series, save_path, title=f"{col} — String Lengths"
                    )
                    interactive = PlotFactory.plot_histogram_interactive(
                        lengths_series,
                        title=f"{col} — String Lengths",
                        annotations=annotation,
                    )

                    plots[col] = {
                        "static": static["path"],
                        "interactive": interactive,
                    }

                except Exception as e:
                    self._log(
                        f"[CategoricalLengthStats] Error processing {col}: {e}",
                        level="debug",
                    )
                    continue

            # Assemble TaskResult
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed string length stats for {len(results)} column(s)."
                    )
                },
                data=results,
                plots=plots,
                metadata={
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Text Summary",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matching_cols + list(excluded.keys())
                    ),
                },
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
