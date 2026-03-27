# dsbf/eda/tasks/detect_duplicates.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Duplicates",
    description="Detects duplicated rows in the dataset.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    tags=["duplicates", "rows"],
    expected_semantic_types=["any"],
)
class DetectDuplicates(BaseTask):
    """
    Detects and counts exact duplicate rows in the dataset.

    A duplicate row is one whose values are identical across all columns to at
    least one other row. The count reported is the number of rows that are
    non-first occurrences - consistent with pandas ``duplicated(keep='first')``.

    For Polars, duplicate count is derived as total rows minus unique rows, which
    is equivalent.

    Output is consumed by the Quality tab (Validity dimension). If duplicates are
    found, an EDA guidance blurb is emitted at the dataset level using the
    ``"__dataset__"`` column key.
    """

    def run(self) -> None:
        """
        Execute duplicate row detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} column(s)",
                "debug",
            )

            if is_polars(df):
                # Polars: unique row count derived from .unique()
                total_rows = df.shape[0]
                unique_rows = df.unique(subset=None, maintain_order=True).shape[0]
                duplicate_count = int(total_rows - unique_rows)
            else:
                duplicate_count = int(df.duplicated().sum())

            self._log(f"    Duplicate row count: {duplicate_count}", "debug")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Found {duplicate_count} duplicate row(s)."},
                data={"duplicate_count": duplicate_count},
                metadata={
                    "suggested_viz_type": "none",
                    "recommended_section": "Duplicates",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            if duplicate_count > 0:
                total_rows: int = df.shape[0] if is_polars(df) else len(df)
                pct: float = duplicate_count / total_rows if total_rows > 0 else 0.0
                pct_str: str = f"{pct:.1%}"

                self.add_guidance(
                    result=self.output,
                    column="__dataset__",
                    phase="eda",
                    level="warn" if pct > 0.01 else "info",
                    title=(
                        f"Duplicate Rows Detected ({duplicate_count} rows, {pct_str})"
                    ),
                    body=(
                        f"{duplicate_count} rows ({pct_str}) are exact duplicates of "
                        f"at least one other row. Duplicate rows can arise from "
                        f"re-ingestion of overlapping data batches, fan-out joins "
                        f"that multiply rows unintentionally, or sampling errors. "
                        f"Verify whether the duplicates are legitimate repeated "
                        f"observations or artefacts of the data pipeline before "
                        f"proceeding with analysis."
                    ),
                    actions=[],
                    metric={
                        "duplicate_count": duplicate_count,
                        "pct_duplicate": round(pct, 4),
                        "total_rows": total_rows,
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
