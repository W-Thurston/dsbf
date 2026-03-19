# dsbf/eda/tasks/summarize_unique.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Summarize Unique Values",
    description="Reports unique value counts per column.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["uniqueness", "summary"],
    expected_semantic_types=["any"],
)
class SummarizeUnique(BaseTask):
    """
    Compute the number of unique values for each column.

    The unique count is computed including nulls for Polars (``n_unique()``
    counts nulls as one distinct value) and excluding nulls for Pandas
    (``nunique()`` excludes nulls by default). This is consistent with how
    each backend defines uniqueness.

    Supports both Pandas and Polars input.
    """

    def run(self) -> None:
        """
        Compute per-column unique counts and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_cols)} column(s)", "debug")

            if is_polars(df):
                self._log(
                    f"    Computing unique values for {len(df.columns)} columns",
                    "debug",
                )
                result: dict[str, int] = {col: df[col].n_unique() for col in df.columns}
            else:
                result = df.nunique().to_dict()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": f"Computed unique counts for {len(result)} columns.",
                },
                data=result,
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Summary",
                    "display_priority": "low",
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
