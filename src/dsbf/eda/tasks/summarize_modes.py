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

            result: dict = {}
            columns_without_mode: list[str] = []

            def collapse(col: str, modes: list) -> None:
                # An all-null column has no mode: record None rather than index [0].
                if not modes:
                    columns_without_mode.append(col)
                result[col] = (modes if len(modes) > 1 else modes[0]) if modes else None

            if is_polars(df):
                for col in df.columns:
                    collapse(col, df[col].drop_nulls().mode().to_list())
            else:
                df_mode = df.mode()
                for col in df_mode.columns:
                    collapse(col, df_mode[col].dropna().tolist())

            message = f"Computed mode(s) for {len(result)} columns."
            if columns_without_mode:
                message += (
                    f" {len(columns_without_mode)} column(s) had no mode "
                    "(all values null)."
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": message},
                data=result,
                metadata={
                    "columns_without_mode": columns_without_mode,
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
