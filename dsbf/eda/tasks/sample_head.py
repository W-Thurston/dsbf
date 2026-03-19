# dsbf/eda/tasks/sample_head.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Sample Head",
    description="Returns the first N rows of the dataset.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["preview"],
    expected_semantic_types=["any"],
)
class SampleHead(BaseTask):
    """
    Returns the first N rows of the dataset for preview.

    Supports both Pandas and Polars DataFrames. The output is serialised to a
    column-oriented dict (``orient="list"``) for JSON portability.

    Configurable parameters (via config["tasks"]["sample_head"]):
        n (int): Number of rows to return. Default: 5
    """

    def run(self) -> None:
        """
        Sample the first N rows and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_cols)} column(s)", "debug")

            n = int(self.get_task_param("n") or 5)
            df_head = df.head(n)
            self._log(f"    Returning first {n} rows", "debug")

            if is_polars(df_head):
                sample = df_head.to_pandas().to_dict(orient="list")
            else:
                sample = df_head.to_dict(orient="list")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Returned first {n} rows."},
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
