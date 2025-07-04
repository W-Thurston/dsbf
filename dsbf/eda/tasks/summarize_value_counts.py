# dsbf/eda/tasks/summarize_value_counts.py

from typing import Any, Dict

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
    domain="core",
    runtime_estimate="fast",
    tags=["categorical", "summary"],
)
class SummarizeValueCounts(BaseTask):
    """
    Computes the top-k most frequent values for each column.

    Converts Polars to Pandas if needed for consistent functionality.
    """

    def run(self) -> None:
        """
        Perform value count summarization on each column, returning the most
        frequent `top_k` values including missing/nulls.
        """
        try:

            # ctx = self.context
            df: Any = self.input_data

            top_k = int(self.get_task_param("top_k") or 5)

            # Convert Polars to Pandas for compatibility with value_counts
            if is_polars(df):
                df = df.to_pandas()
                self._log(
                    "Converting Polars to Pandas for value count computation", "debug"
                )

            result: Dict[str, Dict[Any, int]] = {}

            for col in df.columns:
                try:
                    vc = df[col].value_counts(dropna=False).head(top_k)
                    result[col] = vc.to_dict()
                    self._log(f"Value counts for {col}: {list(vc.index)}", "debug")
                except Exception:
                    continue  # Skip columns that fail (e.g., unhashable types)

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (f"Computed value counts for {len(result)} columns.")
                },
                data=result,
                metadata={"top_k": top_k},
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
