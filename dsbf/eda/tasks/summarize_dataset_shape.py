# dsbf/eda/tasks/summarize_dataset_shape.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Summarize Dataset Shape",
    description="Summarizes dataset dimensions and memory usage.",
    depends_on=["infer_types"],
    stage="raw",
    tags=["overview", "summary"],
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
            df: Any = self.input_data

            # Prefer Polars backend but fallback to Pandas
            if is_polars(df):
                df = df.to_pandas()

            n_rows, n_cols = df.shape
            total_cells = n_rows * n_cols

            null_pct = df.isnull().sum().sum() / total_cells if total_cells else 0.0
            mem_bytes = df.memory_usage(deep=True).sum()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Dataset has {n_rows} rows and {n_cols} columns.",
                data={
                    "num_rows": n_rows,
                    "num_columns": n_cols,
                    "null_cell_percentage": round(null_pct, 4),
                    "approx_memory_MB": round(mem_bytes / 1_048_576, 2),
                },
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during dataset shape summarization: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
