# dsbf/eda/tasks/categorical_length_stats.py

from typing import Any

import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars, is_text_pandas, is_text_polars
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
)
class CategoricalLengthStats(BaseTask):
    """
    Computes string length statistics (mean, min, max) for all text-like categorical
        columns. Supports both Pandas and Polars DataFrames.

    The task inspects each column and, if it's a recognized text-type column,
        computes character length statistics for non-null values.

    Produces a TaskResult containing the summary and per-column stats.
    """

    def run(self) -> None:
        """
        Run the task on input_data and populate self.output as a TaskResult.
        Sets status='success' if the task completes, 'failed' otherwise.
        """

        # ctx = self.context
        df = self.input_data
        results = {}

        try:
            if is_polars(df):
                # Iterate over each column and check if it qualifies as text in Polars
                for col in df.columns:
                    if is_text_polars(df[col]):
                        try:
                            # Convert to Utf8 and compute character lengths
                            lengths = df.select(
                                pl.col(col).cast(pl.Utf8).str.len_chars().alias("len")
                            ).drop_nulls()["len"]

                            # Skip empty results
                            if lengths.len() > 0:
                                results[col] = {
                                    "mean_length": lengths.mean(),
                                    "max_length": lengths.max(),
                                    "min_length": lengths.min(),
                                }
                        except Exception as e:
                            # Gracefully handle and continue on per-column errors
                            self._log(
                                (
                                    f"[CategoricalLengthStats] Error in {col}"
                                    f" (Polars): {e}"
                                ),
                                "debug",
                            )
                            continue
            else:
                # Iterate over Pandas columns and check for text-type columns
                for col in df.columns:
                    if is_text_pandas(df[col]):
                        try:
                            lengths = df[col].dropna().str.len()
                            if len(lengths) > 0:
                                results[col] = {
                                    "mean_length": lengths.mean(),
                                    "max_length": lengths.max(),
                                    "min_length": lengths.min(),
                                }
                        except Exception as e:
                            self._log(
                                (
                                    f"[CategoricalLengthStats] Error in {col}"
                                    f" (Pandas): {e}"
                                ),
                                "debug",
                            )
                            continue

            plots: dict[str, dict[str, Any]] = {}

            for col in results:
                # Compute character lengths
                if is_polars(df):
                    lengths_series = df.select(
                        pl.col(col).cast(pl.Utf8).str.len_chars().alias("len")
                    ).to_pandas()["len"]
                else:
                    lengths_series = df[col].dropna().str.len()

                lengths_series.name = f"{col} length"
                annotation = [
                    f"Min: {results[col]['min_length']:.1f}, "
                    f"Mean: {results[col]['mean_length']:.1f}, "
                    f"Max: {results[col]['max_length']:.1f}"
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

            # Final output
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed string length stats for {len(results)} columns."
                    )
                },
                data=results,
                plots=plots,
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
