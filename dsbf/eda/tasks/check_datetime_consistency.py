# dsbf/eda/tasks/check_datetime_consistency.py

from typing import Any, Dict

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


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
            # Polars fallback — convert to Pandas for datetime ops
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            for col in df.columns:
                try:
                    # Attempt to coerce column to datetime
                    parsed = pd.to_datetime(df[col], errors="coerce")

                    # Compute parseability and monotonicity
                    is_monotonic = (
                        parsed.is_monotonic_increasing or parsed.is_monotonic_decreasing
                    )
                    null_pct = parsed.isnull().mean()

                    results[col] = {
                        "parseable": parsed.notnull().mean() > 0.95,
                        "monotonic": is_monotonic,
                        "null_pct_after_parse": null_pct,
                    }
                except Exception:
                    # Gracefully skip columns that can’t be parsed
                    continue

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Checked datetime consistency for {len(results)} columns.",
                data=results,
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during datetime consistency check: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
