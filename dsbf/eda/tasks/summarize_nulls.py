# dsbf/eda/tasks/summarize_nulls.py

from typing import Any, Dict, List

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task()
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
            df: Any = self.input_data

            if is_polars(df):
                df = df.to_pandas()

            n_rows: int = df.shape[0]

            # Column null counts and percentages
            null_counts: Dict[str, int] = df.isnull().sum().to_dict()
            null_percentages: Dict[str, float] = {
                col: null_counts[col] / n_rows for col in df.columns
            }

            threshold = self.config.get("null_threshold", 0.5)

            high_null_columns: List[str] = [
                col for col, pct in null_percentages.items() if pct >= threshold
            ]

            # Row-wise null pattern frequency (e.g., "101" means null in cols 1 and 3)
            null_mask_df = df.isnull().astype(int)
            null_patterns = null_mask_df.apply(
                lambda row: "".join(row.astype(str)), axis=1
            )
            pattern_counts: Dict[str, int] = null_patterns.value_counts().to_dict()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"{len(high_null_columns)} column(s) have >50% missing values.",
                data={
                    "null_counts": null_counts,
                    "null_percentages": null_percentages,
                    "high_null_columns": high_null_columns,
                    "null_patterns": pattern_counts,
                },
                metadata={"rows": n_rows},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during null summarization: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
