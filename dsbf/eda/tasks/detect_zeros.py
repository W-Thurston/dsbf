# dsbf/eda/tasks/detect_zeros.py

from typing import Literal

import numpy as np

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Zeros",
    description="Flags columns with high zero concentration.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    tags=["zeros", "sparsity"],
    expected_semantic_types=["continuous"],
)
class DetectZeros(BaseTask):
    """
    Detects numeric columns with a high proportion of zero values.

    Computes the count and proportion of zero values per numeric column.
    Columns where the zero proportion exceeds ``flag_threshold`` (default: 0.95)
    are flagged in ``zero_flags``.

    EDA and ML guidance blurbs are emitted for columns where the zero proportion
    ≥ 30% - a lower threshold than the flag threshold to surface structural zeros
    before they become critical.

    The key interpretive question for zero-heavy columns is whether zeros
    represent genuine measurements (zero items sold, zero activity) or missing
    value placeholders. The guidance blurbs surface both interpretations.

    Polars DataFrames are converted to pandas before the zero count computation.

    Configurable parameters (via config["tasks"]["detect_zeros"]):
        flag_threshold (float): Proportion of zeros above which a column is
            flagged as True in ``zero_flags``. Default: 0.95
    """

    def run(self) -> None:
        """
        Execute zero detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'continuous' column(s)",
                "debug",
            )

            flag_threshold = float(self.get_task_param("flag_threshold") or 0.95)

            if is_polars(df):
                df = df.to_pandas()

            if not hasattr(df, "shape"):
                raise ValueError("Input is not a valid dataframe.")  # noqa: TRY301

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

            flagged_count: int = sum(zero_flags.values())

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Flagged {flagged_count} column(s) with high zero counts."
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
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            # Emit guidance at a lower threshold than flag_threshold so
            # findings surface before the column is critically zero-heavy.
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
        """
        Generate EDA and ML guidance for a column with notable zero concentration.

        Args:
            col: Column name.
            pct: Proportion of zero values (0.0 - 1.0).
            count: Absolute count of zero values.

        """
        level: Literal["info", "warn"] = (
            "warn" if pct >= 0.5 else "info"  # noqa: PLR2004
        )
        pct_str: str = f"{pct:.1%}"

        eda_body: str = (
            f"'{col}' has {pct_str} zero values ({count:,} rows). Consider whether "
            f"zeros here represent 'none' or 'absent' - a genuine measurement of "
            f"zero - or whether they are placeholders for missing data. The "
            f"distinction matters: genuine zeros are informative and should be kept, "
            f"while placeholder zeros should be treated as nulls. Check the data "
            f"source or documentation to confirm the intended meaning."
        )

        ml_body: str = (
            f"'{col}' has {pct_str} zero values. If zeros are genuine measurements, "
            f"this column follows a zero-inflated distribution - consider a log1p "
            f"transform or a separate binary indicator (is_zero) to help models "
            f"distinguish the zero mass from the non-zero distribution. If zeros are "
            f"missing-value placeholders, replace them with NaN before fitting to "
            f"avoid silent data leakage into imputation."
        )

        metric: dict[str, float | int] = {
            "zero_pct": round(pct, 4),
            "zero_count": count,
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=(
                f"{'High' if pct >= 0.5 else 'Notable'} "  # noqa: PLR2004
                f"Zero Rate ({pct_str})"
            ),
            body=eda_body.strip(),
            actions=[],
            metric=metric,
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
            metric=metric,
        )
