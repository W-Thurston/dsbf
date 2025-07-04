# dsbf/eda/tasks/check_datetime_consistency.py

from typing import Any, Dict

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Datetime Consistency Check",
    description="Checks for consistency in datetime columns across the dataset.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["datetime", "validation"],
)
class CheckDatetimeConsistency(BaseTask):
    """
    Checks datetime columns for:
        - Whether they can be parsed as dates (parseable)
        - Whether they are monotonic (increasing or decreasing)
        - How many nulls remain after parsing

    Falls back to Pandas even for Polars inputs due to date parsing robustness.
    """

    def run(self) -> None:
        """
        Execute the task using input_data (Polars or Pandas DataFrame).
        Produces a TaskResult with per-column flags and metrics.
        """
        results: Dict[str, Dict[str, Any]] = {}

        try:

            # ctx = self.context
            # Polars fallback — convert to Pandas for datetime ops
            df = self.input_data
            if is_polars(df):
                self._log(
                    "Falling back to Pandas for datetime parsing robustness", "debug"
                )
                df = df.to_pandas()

            for col in df.columns:
                try:
                    # Attempt to coerce column to datetime
                    parsed = pd.to_datetime(df[col], errors="coerce", format="ISO8601")

                    # Compute parseability and monotonicity
                    is_monotonic = bool(
                        parsed.is_monotonic_increasing or parsed.is_monotonic_decreasing
                    )
                    null_pct = parsed.isnull().mean()

                    results[col] = {
                        "parseable": bool(parsed.notnull().mean() > 0.95),
                        "monotonic": is_monotonic,
                        "null_pct_after_parse": null_pct,
                    }
                except Exception:
                    # Gracefully skip columns that can’t be parsed
                    continue

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Checked datetime consistency for {len(results)} columns."
                    )
                },
                data=results,
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
