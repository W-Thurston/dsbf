# dsbf/eda/tasks/summarize_dataset_shape.py

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
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["overview", "summary"],
    expected_semantic_types=["any"],
)
class SummarizeDatasetShape(BaseTask):
    """
    Summarize dataset shape, missingness, and memory usage.

    Computes:
    - Row and column counts
    - Percentage of missing cells across the entire dataset
    - Approximate memory usage in MB (via pandas ``memory_usage(deep=True)``)

    Polars DataFrames are converted to pandas for the memory usage estimation.
    """

    def run(self) -> None:
        """
        Execute shape summarization and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_cols)} column(s)", "debug")

            if is_polars(df):
                self._log(
                    "    Converting Polars to pandas for memory usage estimation",
                    "debug",
                )
                df = df.to_pandas()

            n_rows, n_cols = df.shape
            total_cells = n_rows * n_cols

            null_pct: float = (
                df.isnull().sum().sum() / total_cells if total_cells else 0.0
            )
            mem_bytes = df.memory_usage(deep=True).sum()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Dataset has {n_rows} rows and {n_cols} columns."},
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
