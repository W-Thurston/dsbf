# dsbf/eda/tasks/check_datetime_consistency.py

from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
from pandas import Timestamp

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

if TYPE_CHECKING:
    from pandas._libs.tslibs import NaTType


@register_task(
    display_name="Check Datetime Consistency",
    description="Checks for consistency in datetime columns across the dataset.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    tags=["datetime", "validation"],
    expected_semantic_types=["datetime"],
)
class CheckDatetimeConsistency(BaseTask):
    """
    Validates datetime columns and reports the proportion of parseable values.

    For each column classified as datetime by the type inference task, attempts
    to parse all values and counts failures. A high invalid rate indicates
    heterogeneous formats, timezone issues, or upstream data corruption.

    Supports both Polars (native cast) and Pandas (pd.to_datetime) DataFrames.
    The Polars path uses strict=False casting to avoid exceptions on bad values,
    which is faster than the pandas coerce path for large columns.

    Output is consumed by the frontend Validation section of the Quality tab.
    """

    def run(self) -> None:
        """
        Execute the datetime consistency check and populate self.output.

        Iterates over all datetime-classified columns, attempts parsing,
        and records the count and proportion of invalid values per column.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        df = self.input_data
        results: dict[str, dict[str, Any]] = {}

        try:
            datetime_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(datetime_cols)} datetime column(s)",
                "debug",
            )

            for col in datetime_cols:
                try:
                    if is_polars(df):
                        # strict=False coerces unparseable values to null rather
                        # than raising — equivalent to pandas errors="coerce".
                        parsed = df[col].cast(pl.Datetime, strict=False)
                        total: int = len(parsed)
                        nulls = int(parsed.is_null().sum())
                    else:
                        parsed: NaTType | Timestamp = pd.to_datetime(
                            df[col],
                            errors="coerce",
                        )
                        total = len(parsed)
                        nulls = int(parsed.isna().sum())

                    percent_valid: float = (
                        round(100 * (1.0 - nulls / total), 2) if total > 0 else 0.0
                    )

                    results[col] = {
                        "num_values": total,
                        "num_invalid": nulls,
                        "percent_valid": percent_valid,
                    }

                except Exception as e:  # noqa: BLE001, PERF203
                    self._log(
                        f"    [{self.name}] Error processing column '{col}': {e}",
                        "debug",
                    )
                    continue

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Checked datetime consistency for {len(results)} column(s)."
                    ),
                },
                data=results,
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Validation",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        datetime_cols + list(excluded.keys()),
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
