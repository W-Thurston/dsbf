# dsbf/eda/tasks/sample_tail.py

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
    domain="core",
    runtime_estimate="fast",
    tags=["preview"],
    expected_semantic_types=["any"],
)
class SampleTail(BaseTask):
    """
    Returns the last N rows of the dataset for preview.
    Works with both Pandas and Polars.
    """

    def run(self) -> None:
        try:
            # ctx = self.context
            df = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(matched_col)} column(s)", "debug")

            n = self.get_task_param("n")
            if n is None:
                n = 5
            else:
                n = int(n)

            if n == 0:
                df_tail = df.head(0)
            else:
                df_tail = df.tail(n)

            if is_polars(df_tail):
                result = df_tail.to_pandas().to_dict(orient="list")
            else:
                result = df_tail.to_dict(orient="list")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": (f"Returned last {n} rows.")},
                data={"sample": result},
                metadata={
                    "n": n,
                    "suggested_viz_type": "table",
                    "recommended_section": "Preview",
                    "display_priority": "low",
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
