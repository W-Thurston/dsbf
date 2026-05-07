# dsbf/eda/tasks/summarize_dataset_shape.py


from typing import TYPE_CHECKING

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result

if TYPE_CHECKING:
    import pandas as pd


@register_task(
    display_name="Summarize Dataset Shape",
    description=(
        "Summarizes dataset dimensions, memory usage, and per-column memory breakdown."
    ),
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
    - Approximate total memory usage in MB
    - Per-column memory usage in bytes and MB

    The per-column breakdown feeds ``suggest_dtype_optimizations``, which reads
    from this task's output and recommends dtype downcasts for columns with
    unnecessarily large storage types. The shape task describes what *is*;
    the optimization task recommends what *could be smaller*.

    Polars DataFrames are converted to pandas for memory estimation since
    ``memory_usage(deep=True)`` is a pandas API.
    """

    def run(self) -> None:
        """
        Execute shape summarization and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run()

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No columns found — dataset shape summary skipped.",
                    excluded,
                )
                return

            n_rows, n_cols = df.shape
            total_cells = n_rows * n_cols

            null_pct: float = (
                df.isna().sum().sum() / total_cells if total_cells else 0.0
            )

            # per-column memory - deep=True includes referenced objects (e.g. strings)
            col_memory: pd.Series = df.memory_usage(deep=True)
            # pandas includes an "Index" entry; exclude it
            col_memory_bytes: dict[str, int] = {
                col: int(col_memory[col])
                for col in df.columns
                if col in col_memory.index
            }
            total_mem_bytes: int = int(col_memory.sum())

            self._log(
                f"    Total memory: {total_mem_bytes / 1_048_576:.3f} MB "
                f"across {n_cols} columns",
                "debug",
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Dataset has {n_rows} rows and {n_cols} columns."},
                data={
                    "num_rows": n_rows,
                    "num_columns": n_cols,
                    "null_cell_percentage": round(null_pct, 4),
                    "approx_memory_MB": round(total_mem_bytes / 1_048_576, 2),
                    # Per-column breakdown - consumed by suggest_dtype_optimizations
                    # and available to the Overview tab for column-level memory display.
                    "column_memory_bytes": col_memory_bytes,
                    "column_memory_MB": {
                        col: round(b / 1_048_576, 4)
                        for col, b in col_memory_bytes.items()
                    },
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
