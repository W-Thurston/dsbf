# dsbf/eda/tasks/summarize_nulls.py

from typing import Any, Dict, List

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Summarize Nulls",
    description="Reports null value counts per column.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["nulls", "missing"],
)
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

            # ctx = self.context
            df: Any = self.input_data

            null_threshold = float(self.get_task_param("null_threshold") or 0.5)

            if is_polars(df):
                df = df.to_pandas()

            n_rows: int = df.shape[0]

            # Column null counts and percentages
            null_counts: Dict[str, int] = df.isnull().sum().to_dict()
            null_percentages: Dict[str, float] = {
                col: null_counts[col] / n_rows for col in df.columns
            }

            high_null_columns: List[str] = [
                col for col, pct in null_percentages.items() if pct >= null_threshold
            ]
            self._log(
                f"Detected {len(high_null_columns)} columns with >50% nulls", "debug"
            )

            # Row-wise null pattern frequency (e.g., "101" means null in cols 1 and 3)
            null_mask_df = df.isnull().astype(int)
            null_patterns = null_mask_df.apply(
                lambda row: "".join(row.astype(str)), axis=1
            )
            pattern_counts: Dict[str, int] = null_patterns.value_counts().to_dict()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"{len(high_null_columns)} column(s) have >50% missing values."
                    )
                },
                data={
                    "null_counts": null_counts,
                    "null_percentages": null_percentages,
                    "high_null_columns": high_null_columns,
                    "null_patterns": pattern_counts,
                },
                metadata={"rows": n_rows},
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
