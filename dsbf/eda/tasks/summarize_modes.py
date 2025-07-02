# dsbf/eda/tasks/summarize_modes.py

from typing import Any

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
    domain="core",
    runtime_estimate="fast",
    tags=["modes", "summary"],
)
class SummarizeModes(BaseTask):
    """
    Summarizes the mode(s) — the most frequent value(s) — for each column.

    Handles both Polars and Pandas backends, returning a dictionary where:
    - Each key is a column name.
    - Each value is either the mode (single value) or a list of modes if multimodal.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            if is_polars(df):
                result = {
                    col: (
                        values
                        if len(values := df[col].mode().to_list()) > 1
                        else values[0]
                    )
                    for col in df.columns
                }
            else:
                df_mode = df.mode()
                result = {
                    col: (
                        col_modes.dropna().tolist()
                        if len(col_modes := df_mode[col]) > 1
                        else col_modes.iloc[0]
                    )
                    for col in df_mode.columns
                }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": (f"Computed mode(s) for {len(result)} columns.")},
                data=result,
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
