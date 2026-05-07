# dsbf/eda/tasks/detect_mixed_type_columns.py

from collections import Counter, defaultdict

import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    name="detect_mixed_type_columns",
    display_name="Detect Mixed-Type Columns",
    description=(
        "Flags columns that contain multiple Python data types (e.g., str + float)."
    ),
    depends_on=["infer_types"],
    tags=["type", "format", "anomaly"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    expected_semantic_types=["any"],
)
class DetectMixedTypeColumns(BaseTask):
    """
    Detects columns that contain more than one Python data type.

    Mixed-type columns arise when a string column contains a mix of strings
    and numeric values, or when a column has been improperly coerced. They
    are stored as ``object`` dtype in pandas or ``pl.Object`` in Polars and
    can silently corrupt aggregations, sorting, and model ingestion.

    Detection is based on inspecting the Python types of individual values
    after iterating the column. Only columns with at least two distinct types
    that each represent ≥ ``min_ratio`` of non-null values are flagged.

    For Polars, only ``pl.Object`` dtype columns are inspected - strictly typed
    columns (``String``, ``Int64``, etc.) cannot contain mixed types by definition.
    For Pandas, only ``object`` dtype columns are inspected.

    Configurable parameters (via config["tasks"]["detect_mixed_type_columns"]):
        min_ratio (float): Minimum proportion of non-null values a minority type
            must represent to trigger a flag. Default: 0.05
        ignore_null_type (bool): Exclude ``NoneType`` from type counting.
            Default: True
    """

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """
        Execute mixed-type column detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run_native()

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No categorical columns found — mixed type detection skipped.",
                    excluded,
                )
                return

            min_ratio = float(self.get_task_param("min_ratio") or 0.05)
            ignore_null_type = bool(self.get_task_param("ignore_null_type") or True)

            flagged_columns: list[str] = []
            details: dict = {}
            recommendations: list[str] = []

            for col in df.columns:
                # Polars: only pl.Object columns can contain mixed Python types.
                # Strictly typed columns (String, Int64, etc.) cannot.
                if is_polars(df):
                    if df[col].dtype != pl.Object:
                        continue
                # Pandas: only object dtype can contain mixed types.
                elif df[col].dtype != object:
                    continue

                try:
                    values = df[col].to_numpy() if is_polars(df) else df[col].values
                except Exception:  # noqa: BLE001, S112
                    continue

                type_counter: Counter = Counter()
                for v in values:
                    if v is None:
                        if not ignore_null_type:
                            type_counter["NoneType"] += 1
                    else:
                        type_counter[type(v).__name__] += 1

                if len(type_counter) <= 1:
                    continue

                total: int = sum(type_counter.values())
                if total == 0:
                    continue

                minority_types: dict = {
                    t: count
                    for t, count in type_counter.items()
                    if count / total >= min_ratio
                    and count != max(type_counter.values())
                }

                if not minority_types:
                    continue

                samples_by_type: dict = defaultdict(list)
                for v in values:
                    if v is None and ignore_null_type:
                        continue
                    tname: str = type(v).__name__
                    if tname in minority_types and len(samples_by_type[tname]) < 5:
                        samples_by_type[tname].append(repr(v))

                flagged_columns.append(col)
                details[col] = {
                    "type_counts": dict(type_counter),
                    "sample_values": dict(samples_by_type),
                }
                recommendations.append(
                    f"Column '{col}' contains multiple data types "
                    f"({', '.join(type_counter)}). "
                    "Consider cleaning or coercing values.",
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "num_mixed_type_columns": len(flagged_columns),
                    "columns": flagged_columns,
                },
                data=details,
                recommendations=recommendations,
                metadata={
                    "suggested_viz_type": "none",
                    "recommended_section": "Validation",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col in flagged_columns:
                self._attach_guidance(col, details[col])

            # ML impact scoring
            if self.get_engine_param("enable_impact_scoring", True) and flagged_columns:
                top_col: str = flagged_columns[0]
                type_info = details[top_col].get("type_counts", {})
                tip: str | None = get_recommendation_tip(
                    self.name,
                    {"type_counts": type_info},
                )
                self.set_ml_signals(
                    result=self.output,
                    score=0.6,
                    tags=["transform"],
                    recommendation=tip
                    or (
                        f"Column '{top_col}' contains mixed types "
                        f"({', '.join(type_info)}). Consider coercing to a "
                        "single type or cleaning inconsistent entries."
                    ),
                )
                self.output.summary["column"] = top_col

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, col_details: dict) -> None:
        """
        Generate EDA guidance for a column with mixed value types.

        Args:
            col: Column name.
            col_details: Detection result dict containing ``type_counts``
                and ``sample_values``.

        """
        type_counts = col_details["type_counts"]
        type_str: str = ", ".join(
            f"{t} ({c})" for t, c in sorted(type_counts.items(), key=lambda x: -x[1])
        )

        eda_body: str = (
            f"'{col}' contains more than one Python type: {type_str}. "
            f"Mixed types in a column indicate inconsistent data entry, "
            f"an incomplete type coercion step, or values that were parsed "
            f"differently across batches or sources. Aggregations (sum, mean) "
            f"on this column will silently fail or produce incorrect results. "
            f"Inspect the minority-type values to determine whether they are "
            f"errors or legitimate edge cases."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="warn",
            title=f"Mixed Value Types ({len(type_counts)} types detected)",
            body=eda_body.strip(),
            actions=[],
            metric={"type_counts": type_counts},
        )
