# dsbf/eda/tasks/detect_zeros.py

from typing import Any, Literal

import numpy as np

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Zeros",
    description="Flags columns or rows with high zero concentration.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["zeros", "sparsity"],
    expected_semantic_types=["continuous"],
)
class DetectZeros(BaseTask):
    """
    Detects numeric columns with a high proportion of zero values.

    Flags columns where zeros exceed a specified threshold.
    """

    def run(self) -> None:
        try:
            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_col)} 'continuous' column(s)",
                "debug",
            )

            flag_threshold = float(self.get_task_param("flag_threshold") or 0.95)

            if is_polars(df):
                df = df.to_pandas()

            if not hasattr(df, "shape"):
                raise ValueError("Input is not a valid dataframe.")

            n_rows = df.shape[0]
            zero_counts: dict[str, int] = {}
            zero_percentages: dict[str, float] = {}
            zero_flags: dict[str, bool] = {}

            numeric_df = df.select_dtypes(include=[np.number])

            for col in numeric_df.columns:
                count = int((numeric_df[col] == 0).sum())
                pct = count / n_rows
                zero_counts[col] = count
                zero_percentages[col] = pct
                zero_flags[col] = pct > flag_threshold

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Flagged {sum(zero_flags.values())}"
                        f"column(s) with high zero counts."
                    ),
                },
                data={
                    "zero_counts": zero_counts,
                    "zero_percentages": zero_percentages,
                    "zero_flags": zero_flags,
                },
                metadata={
                    "threshold_pct": flag_threshold,
                    "total_rows": n_rows,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Sparsity",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys()),
                    ),
                },
                plots={},
            )

            # Generate guidance for any column with notable zero concentration.
            # Threshold is lower than flag_threshold - guidance starts at 30%
            # to match what a reviewer would find worth investigating.
            guidance_threshold = 0.30
            for col, pct in zero_percentages.items():
                if pct >= guidance_threshold:
                    self._attach_guidance(col, pct, zero_counts[col])

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, pct: float, count: int) -> None:
        """Generate EDA + ML guidance for a column with notable zero concentration."""
        level: Literal["info", "warn"] = "warn" if pct >= 0.5 else "info"
        pct_str: str = f"{pct:.1%}"

        eda_body: str = (
            f"{col} has {pct_str} zero values ({count:,} rows). "
            f"Consider whether zeros here represent 'none' or 'absent' - a genuine "
            f"measurement of zero - or whether they are placeholders for missing data. "
            f"The distinction matters: genuine zeros are informative and should be "
            f"kept, while placeholder zeros should be treated as nulls. "
            f"Check the data source or documentation to confirm the intended meaning."
        )

        ml_body: str = (
            f"{col} has {pct_str} zero values. If zeros are genuine measurements, "
            f"this column follows a zero-inflated distribution - consider a log1p "
            f"transform or a separate binary indicator (is_zero) to help models "
            f"distinguish the zero mass from the non-zero distribution. "
            f"If zeros are missing-value placeholders, replace them with NaN before "
            f"fitting to avoid silent data leakage into imputation."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=f"{'High' if pct >= 0.5 else 'Notable'} Zero Rate ({pct_str})",
            body=eda_body.strip(),
            actions=[],
            metric={"zero_pct": round(pct, 4), "zero_count": count},
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title=f"Zero-Inflated Column ({pct_str} zeros)",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "transform",
                    "method": "log1p",
                    "column": col,
                    "condition": "zeros are genuine",
                },
                {
                    "action": "add_indicator",
                    "method": "is_zero",
                    "column": col,
                    "condition": "zeros are genuine",
                },
                {
                    "action": "replace_with_nan",
                    "column": col,
                    "condition": "zeros are missing-value placeholders",
                },
            ],
            metric={"zero_pct": round(pct, 4), "zero_count": count},
        )
