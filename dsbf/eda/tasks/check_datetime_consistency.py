# dsbf/eda/tasks/check_datetime_consistency.py

from typing import Any, Dict

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result


@register_task(
    display_name="Check Datetime Consistency",
    description="Checks for consistency in datetime columns across the dataset.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["datetime", "validation"],
    expected_semantic_types=["datetime"],
)
class CheckDatetimeConsistency(BaseTask):
    """
    Checks for datetime columns and flags rows where parsing
    fails or yields inconsistent formats.
    Intended to surface dirty or heterogeneous timestamp data.
    """

    def run(self) -> None:
        """
        Execute the datetime consistency check.
        For each datetime column, parse and check for nulls (parsing failures).
        Reports number and percentage of inconsistent values.
        """
        df = self.input_data
        results: Dict[str, Dict[str, Any]] = {}

        try:
            # Step 1: Select datetime-like columns via semantic typing
            datetime_cols, excluded = self.get_columns_by_intent()

            for col in datetime_cols:
                try:
                    # Step 2: Attempt datetime parsing (fallback to pandas)
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    total = len(parsed)
                    nulls = parsed.isna().sum()
                    consistency = 1.0 - (nulls / total) if total > 0 else 0.0

                    results[col] = {
                        "num_values": total,
                        "num_invalid": nulls,
                        "percent_valid": round(100 * consistency, 2),
                    }

                except Exception as e:
                    self._log(
                        f"[CheckDatetimeConsistency] Error in column {col}: {e}",
                        "debug",
                    )
                    continue

            # Step 3: Package results in a TaskResult
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Checked datetime consistency for {len(results)} column(s)."
                    )
                },
                data=results,
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Validation",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        datetime_cols + list(excluded.keys())
                    ),
                },
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
