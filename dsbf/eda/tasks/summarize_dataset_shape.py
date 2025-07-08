# dsbf/eda/tasks/summarize_dataset_shape.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Summarize Dataset Shape",
    description="Summarizes dataset dimensions and memory usage.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["overview", "summary"],
    expected_semantic_types=["any"],
)
class SummarizeDatasetShape(BaseTask):
    """
    Summarizes dataset shape and structure:
    - Row and column counts
    - Approximate memory usage in MB
    - Percentage of missing cells
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(matched_col)} column(s)", "debug")

            # Prefer Polars backend but fallback to Pandas
            if is_polars(df):
                df = df.to_pandas()
                self._log(
                    "Converting Polars to Pandas for memory usage estimation", "debug"
                )

            n_rows, n_cols = df.shape
            total_cells = n_rows * n_cols

            null_pct = df.isnull().sum().sum() / total_cells if total_cells else 0.0
            mem_bytes = df.memory_usage(deep=True).sum()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (f"Dataset has {n_rows} rows and {n_cols} columns.")
                },
                data={
                    "num_rows": n_rows,
                    "num_columns": n_cols,
                    "null_cell_percentage": round(null_pct, 4),
                    "approx_memory_MB": round(mem_bytes / 1_048_576, 2),
                },
                metadata={
                    "suggested_viz_type": "summary",
                    "recommended_section": "Overview",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
