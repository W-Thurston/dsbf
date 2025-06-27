# dsbf/eda/tasks/summarize_value_counts.py

from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task()
class SummarizeValueCounts(BaseTask):
    """
    Computes the top-k most frequent values for each column.

    Converts Polars to Pandas if needed for consistent functionality.
    """

    def __init__(self, top_k: int = 5) -> None:
        """
        Initialize the task with configurable top_k.

        Args:
            top_k (int): Number of top values to include per column.
        """
        super().__init__()
        self.top_k = top_k

    def run(self) -> None:
        """
        Perform value count summarization on each column, returning the most
        frequent `top_k` values including missing/nulls.
        """
        try:
            df: Any = self.input_data

            # Convert Polars to Pandas for compatibility with value_counts
            if is_polars(df):
                df = df.to_pandas()

            result: Dict[str, Dict[Any, int]] = {}

            for col in df.columns:
                try:
                    vc = df[col].value_counts(dropna=False).head(self.top_k)
                    result[col] = vc.to_dict()
                except Exception:
                    continue  # Skip columns that fail (e.g., unhashable types)

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Computed value counts for {len(result)} columns.",
                data=result,
                metadata={"top_k": self.top_k},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during value count summarization: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
