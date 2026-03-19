# dsbf/eda/tasks/summarize_value_counts.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Summarize Value Counts",
    description="Lists value frequencies for selected columns.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["categorical", "summary"],
    expected_semantic_types=["any"],
)
class SummarizeValueCounts(BaseTask):
    """
    Compute the top-k most frequent values for each column.

    For each column in the DataFrame, returns the ``top_k`` most frequent
    values including nulls (``dropna=False``), stored as a column-keyed dict
    of ``{value: count}`` pairs.

    Polars DataFrames are converted to pandas before processing since pandas
    ``value_counts()`` is used for consistent null handling.

    Configurable parameters (via config["tasks"]["summarize_value_counts"]):
        top_k (int): Number of most frequent values to return per column.
            Default: 5
    """

    def run(self) -> None:
        """
        Compute top-k value counts per column and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_cols)} column(s)", "debug")

            top_k = int(self.get_task_param("top_k") or 5)

            if is_polars(df):
                self._log(
                    "    Converting Polars to pandas for value count computation",
                    "debug",
                )
                df = df.to_pandas()

            result: dict[str, dict[Any, int]] = {}

            for col in df.columns:
                try:
                    vc = df[col].value_counts(dropna=False).head(top_k)
                    result[col] = vc.to_dict()
                except Exception:  # noqa: BLE001, PERF203, S112
                    continue  # Skip columns with unhashable or incomparable types

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": f"Computed value counts for {len(result)} columns.",
                },
                data=result,
                metadata={
                    "top_k": top_k,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Summary",
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
