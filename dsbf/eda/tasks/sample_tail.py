# dsbf/eda/tasks/sample_tail.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Sample Tail",
    description="Returns the last N rows of the dataset.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["preview"],
    expected_semantic_types=["any"],
)
class SampleTail(BaseTask):
    """
    Returns the last N rows of the dataset for preview.

    Supports both Pandas and Polars DataFrames. The output is serialised to a
    column-oriented dict (``orient="list"``) for JSON portability.

    When ``n=0`` is configured, returns an empty sample rather than the full
    dataset tail - this is intentional behavior for callers that want schema
    inspection without row data.

    Configurable parameters (via config["tasks"]["sample_tail"]):
        n (int): Number of rows to return. Default: 5
    """

    def run(self) -> None:
        """
        Sample the last N rows and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run_native()

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No eligible columns found — sample tail skipped.",
                    excluded,
                )
                return

            n_raw: Any | None = self.get_task_param("n")
            n: int = int(n_raw) if n_raw is not None else 5

            # n=0 returns an empty sample rather than the full tail - see docstring.
            df_tail = df.head(0) if n == 0 else df.tail(n)
            self._log(f"    Returning last {n} rows", "debug")

            if is_polars(df_tail):
                sample = df_tail.to_pandas().to_dict(orient="list")
            else:
                sample = df_tail.to_dict(orient="list")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Returned last {n} rows."},
                data={"sample": sample},
                metadata={
                    "n": n,
                    "suggested_viz_type": "table",
                    "recommended_section": "Preview",
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
