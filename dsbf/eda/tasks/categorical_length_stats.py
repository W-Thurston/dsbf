# dsbf/eda/tasks/categorical_length_stats.py

import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Categorical Length Stats",
    description="Computes string length statistics for text-like categorical columns.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    tags=["categorical", "text", "stats"],
    expected_semantic_types=["categorical", "text"],
)
class CategoricalLengthStats(BaseTask):
    """
    Computes string length statistics for all text-like categorical columns.

    For each matched column, computes mean, min, and max character length across
    all non-null values. Supports both Polars and Pandas DataFrames, preferring
    native Polars operations for performance.

    Output is consumed by the frontend Distributions tab to populate the
    TextLengthCard component per column.
    """

    def run(self) -> None:
        """
        Execute the task and populate self.output with a TaskResult.

        Selects categorical and text columns via semantic intent, computes
        string length statistics for each, and assembles the result.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        df = self.input_data
        results: dict[str, dict[str, int | float]] = {}

        try:
            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} ['categorical', 'text'] column(s)",
                "debug",
            )

            for col in matched_cols:
                try:
                    if is_polars(df):
                        # Cast to String to handle mixed or enum types, then
                        # compute character-level lengths (not byte lengths).
                        lengths = df.select(
                            pl.col(col).cast(pl.String).str.len_chars().alias("len"),
                        ).drop_nulls()["len"]

                        if lengths.len() == 0:
                            continue

                        results[col] = {
                            "mean_length": float(lengths.mean()),
                            "max_length": int(lengths.max()),
                            "min_length": int(lengths.min()),
                        }
                    else:
                        lengths = df[col].dropna().astype(str).str.len()

                        if len(lengths) == 0:
                            continue

                        results[col] = {
                            "mean_length": float(lengths.mean()),
                            "max_length": int(lengths.max()),
                            "min_length": int(lengths.min()),
                        }

                except Exception as e:  # noqa: BLE001
                    self._log(
                        f"    [{self.name}] Error processing column '{col}': {e}",
                        level="debug",
                    )
                    continue

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed string length stats for {len(results)} column(s)."
                    ),
                },
                data=results,
                metadata={
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Text Summary",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
