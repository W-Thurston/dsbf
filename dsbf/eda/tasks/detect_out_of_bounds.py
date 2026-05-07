# dsbf/eda/tasks/detect_out_of_bounds.py


from typing import TYPE_CHECKING

import numpy as np

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result

if TYPE_CHECKING:
    from pandas import DataFrame, Series


@register_task(
    display_name="Detect Out of Bounds",
    description="Detects numeric values outside expected or logical ranges.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    tags=["bounds", "validation"],
    expected_semantic_types=["continuous"],
)
class DetectOutOfBounds(BaseTask):
    """
    Detects numeric columns with values outside expected domain-specific bounds.

    Checks each numeric column name against a configurable bounds dictionary.
    Only columns whose names appear in the bounds dict are checked - all others
    are silently skipped.

    The default bounds cover common column name patterns: ``age``, ``temperature``,
    ``percent``, and ``score``. Custom bounds can be provided via task config.

    Polars DataFrames are converted to pandas for the bounds check since the
    filtering logic uses pandas boolean indexing.

    For each flagged column, both EDA and ML guidance blurbs are emitted describing
    the violation and recommended remediation.

    Configurable parameters (via config["tasks"]["detect_out_of_bounds"]):
        custom_bounds (dict): Mapping of column name → [lower, upper] list.
            Default: {"age": [0, 120], "temperature": [-100, 150],
                      "percent": [0, 100], "score": [0, 1]}

    Note: Bounds must be specified as YAML lists [min, max] in config, not Python
    tuples (min, max). YAML does not support tuple syntax and will misparse them.
    """

    def run(self) -> None:
        """
        Execute out-of-bounds detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'continuous' column(s)",
                "debug",
            )

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No continuous columns found — out-of-bounds detection skipped.",
                    excluded,
                )
                return

            bounds: dict[str, tuple[float, float]] = dict(
                self.get_task_param("custom_bounds")
                or {
                    "age": (0, 120),
                    "temperature": (-100, 150),
                    "percent": (0, 100),
                    "score": (0, 1),
                },
            )

            df: DataFrame = self.get_dataframe_pandas()

            flagged: dict[str, dict] = {}

            for col in df.select_dtypes(include=np.number).columns:
                if col not in bounds:
                    continue
                # Cast to float defensively — bounds loaded from YAML config
                # are parsed as lists of ints/floats with the correct syntax
                # (e.g. [0, 120]), but cast here as belt-and-suspenders against
                # any future config variations.
                lower = float(bounds[col][0])
                upper = float(bounds[col][1])
                series: Series = df[col].dropna()
                violations = series[(series < lower) | (series > upper)]

                if not violations.empty:
                    flagged[col] = {
                        "count": int(violations.count()),
                        "min_violation": float(violations.min()),
                        "max_violation": float(violations.max()),
                        "expected_range": (float(lower), float(upper)),
                    }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Detected {len(flagged)} column(s) with out-of-bounds values."
                    ),
                },
                data=flagged,
                metadata={
                    "rule_columns": list(bounds.keys()),
                    "suggested_viz_type": "none",
                    "recommended_section": "Validation",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, violation in flagged.items():
                self._attach_guidance(col, violation)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, violation: dict) -> None:
        """
        Generate EDA and ML guidance for a column with out-of-bounds values.

        Args:
            col: Column name.
            violation: Violation dict containing ``count``, ``min_violation``,
                ``max_violation``, and ``expected_range``.

        """
        count = violation["count"]
        min_v = violation["min_violation"]
        max_v = violation["max_violation"]
        lo, hi = violation["expected_range"]

        if min_v < lo and max_v > hi:
            breach_desc: str = f"below {lo} and above {hi}"
        elif min_v < lo:
            breach_desc = f"below the minimum of {lo} (lowest seen: {min_v})"
        else:
            breach_desc = f"above the maximum of {hi} (highest seen: {max_v})"

        eda_body: str = (
            f"'{col}' has {count:,} value(s) outside the expected range "
            f"[{lo}, {hi}]: {breach_desc}. Out-of-bounds values may indicate "
            f"data entry errors, unit mismatches (e.g. a temperature recorded in "
            f"Fahrenheit in a Celsius column), or genuine edge cases outside the "
            f"defined domain. Investigate the source of these values before "
            f"treating them as valid."
        )

        ml_body: str = (
            f"'{col}' has {count:,} value(s) outside [{lo}, {hi}]. If these are "
            f"errors, cap or remove them before modeling - they will distort learned "
            f"boundaries and make the model brittle at the edges of the distribution. "
            f"If they are genuine, confirm the model will encounter similar values "
            f"at inference time."
        )

        metric: dict = {
            "violation_count": count,
            "min_violation": min_v,
            "max_violation": max_v,
            "expected_range": [lo, hi],
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="warn",
            title=f"Out-of-Bounds Values ({count:,} rows)",
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="warn",
            title="Out-of-Bounds Values - Validate Before Modeling",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "investigate",
                    "column": col,
                    "detail": (
                        f"Confirm whether values outside [{lo}, {hi}] are "
                        "errors or genuine observations"
                    ),
                },
                {
                    "action": "winsorize",
                    "column": col,
                    "detail": f"Cap to [{lo}, {hi}] if violations are errors",
                },
                {
                    "action": "remove_rows",
                    "column": col,
                    "detail": "Remove violating rows if confirmed data entry errors",
                },
            ],
            metric=metric,
        )
