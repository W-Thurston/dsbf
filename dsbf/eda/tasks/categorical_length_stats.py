# dsbf/eda/tasks/categorical_length_stats.py

import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars, is_text_pandas, is_text_polars


@register_task(
    display_name="Categorical Length Stats",
    description="Computes string length statistics for text-like categorical columns.",
    depends_on=["infer_types"],
    stage="cleaned",
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
                            print(
                                f"[CategoricalLengthStats] Error in {col} (Polars): {e}"
                            )
                            continue
            else:
                # Iterate over Pandas columns and check for text-type columns
                for col in df.columns:
                    if is_text_pandas(df[col]):
                        try:
                            lengths = df[col].dropna().astype(str).map(len)
                            if len(lengths) > 0:
                                results[col] = {
                                    "mean_length": lengths.mean(),
                                    "max_length": lengths.max(),
                                    "min_length": lengths.min(),
                                }
                        except Exception as e:
                            print(
                                f"[CategoricalLengthStats] Error in {col} (Pandas): {e}"
                            )
                            continue

            # Final output
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Computed string length stats for {len(results)} columns.",
                data=results,
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=str(e),
                data=None,
                metadata={"exception": type(e).__name__},
            )
