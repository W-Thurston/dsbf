# dsbf/eda/tasks/summarize_modes.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Summarize Modes",
    description="Finds most frequent values per column.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["modes", "summary"],
    expected_semantic_types=["any"],
)
class SummarizeModes(BaseTask):
    """
    Summarize the mode(s) - most frequent value(s) - for each column.

    For columns with a single mode, stores the scalar value. For columns with
    multiple equally-frequent modes, stores a list.

    Supports both Polars and Pandas DataFrames.
    """

    def run(self) -> None:
        """
        Compute per-column modes and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run_native()

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No categorical columns found — mode summary skipped.",
                    excluded,
                )
                return

            if is_polars(df):
                result: dict = {}
                for col in df.columns:
                    modes = df[col].mode().to_list()
                    result[col] = modes if len(modes) > 1 else modes[0]
            else:
                df_mode = df.mode()
                result = {}
                for col in df_mode.columns:
                    col_modes = df_mode[col].dropna().tolist()
                    result[col] = col_modes if len(col_modes) > 1 else col_modes[0]

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Computed mode(s) for {len(result)} columns."},
                data=result,
                metadata={
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
