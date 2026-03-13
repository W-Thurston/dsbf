# dsbf/eda/tasks/detect_outliers.py

from typing import Any

import numpy as np

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    display_name="Detect Outliers",
    description="Uses statistical heuristics to flag outlier values.",
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="moderate",
    tags=["outliers", "numeric"],
    expected_semantic_types=["continuous"],
)
class DetectOutliers(BaseTask):
    """
    Detects numeric outliers using the IQR method. Flags columns exceeding
    a proportion threshold of outliers.
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

            method = str(self.get_task_param("method") or "iqr")
            flag_threshold = float(self.get_task_param("flag_threshold") or 0.01)

            if is_polars(df):
                df = df.to_pandas()

            if not hasattr(df, "shape"):
                raise ValueError("Input is not a valid dataframe.")

            n_rows = df.shape[0]
            outlier_counts: dict[str, int] = {}
            outlier_flags: dict[str, bool] = {}
            outlier_rows: dict[str, list[int]] = {}

            numeric_df = df.select_dtypes(include=[np.number])
            iqr_bounds: dict[str, dict[str, float]] = {}

            for col in numeric_df.columns:
                series = numeric_df[col].dropna()

                if series.empty:
                    self._log(f"    {col} skipped: empty after dropna()", "debug")
                    continue

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_mask = (series < lower) | (series > upper)
                indices = series[outlier_mask].index.tolist()

                outlier_counts[col] = len(indices)
                outlier_rows[col] = indices
                outlier_flags[col] = len(indices) > flag_threshold * n_rows
                iqr_bounds[col] = {
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower": float(lower),
                    "upper": float(upper),
                }

            flagged_cols: list[str] = [
                col for col, flagged in outlier_flags.items() if flagged
            ]

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (f"Detected outliers in {len(flagged_cols)} column(s)."),
                },
                data={
                    "outlier_counts": outlier_counts,
                    "outlier_flags": outlier_flags,
                    "outlier_rows": outlier_rows,
                },
                metadata={
                    "method": method,
                    "threshold_pct": flag_threshold,
                    "total_rows": n_rows,
                    "suggested_viz_type": "boxplot",
                    "recommended_section": "Outliers",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys()),
                    ),
                },
                plots={},
            )

            # Generate guidance for every column with at least one outlier
            for col, count in outlier_counts.items():
                if count > 0:
                    pct = count / n_rows
                    bounds: dict[str, float] = iqr_bounds.get(col, {})
                    self._attach_guidance(col, count, pct, n_rows, bounds)

            # Apply ML scoring to self.output
            if self.get_engine_param("enable_impact_scoring", True) and flagged_cols:
                col: str = flagged_cols[0]
                n_outliers: int = outlier_counts[col]
                tip: str | None = get_recommendation_tip(
                    self.name, {"n_outliers": n_outliers}
                )
                self.set_ml_signals(
                    result=self.output,
                    score=0.7,
                    tags=["monitor", "transform"],
                    recommendation=tip
                    or (
                        f"Column '{col}' contains {n_outliers} statistical outliers. "
                        "Consider log-transforming, winsorizing,"
                        " or using robust models."
                    ),
                )
                self.output.summary["column"] = col

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(
        self,
        col: str,
        count: int,
        pct: float,
        n_rows: int,
        bounds: dict[str, float],
    ) -> None:
        """Generate EDA + ML guidance for a column with IQR-detected outliers."""
        pct_str: str = f"{pct:.1%}"
        lower: float | None = bounds.get("lower")
        upper: float | None = bounds.get("upper")
        bounds_desc: str = (
            f"outside the IQR fence [{lower:.4g}, {upper:.4g}]"
            if lower is not None and upper is not None
            else "outside the IQR fence"
        )

        if pct >= 0.15:
            level = "error"
            severity = "severe"
        elif pct >= 0.05:
            level = "warn"
            severity = "notable"
        else:
            level = "info"
            severity = "low-level"

        eda_body: str = (
            f"{col} has {count:,} values ({pct_str} of {n_rows:,} rows) {bounds_desc}. "
            f"{
                'At this rate, calling them outliers is misleading - more than one in '
                'seven rows falls outside the normal range, which suggests the '
                'distribution is simply heavy-tailed or multimodal rather than'
                ' contaminated by a few anomalous points.'
                if pct >= 0.15
                else ''
            }"
            f"{
                'This is a notable minority - enough to materially affect mean-based '
                'statistics but small enough that these could be genuine rare events.'
                if 0.05 <= pct < 0.15
                else ''
            }"
            f"{
                'A small number of values sit unusually far from the bulk of the '
                'distribution.'
                if pct < 0.05
                else ''
            } "
            "Check whether these extreme values are plausible for the domain, arise "
            "from measurement or entry errors, or represent a distinct sub-population "
            "worth analysing separately."
        )

        ml_body: str = (
            f"{col} has {count:,} outlier values ({pct_str}) by the IQR method. "
            "Linear models (regression, SVM with RBF kernel) and distance-based "
            "methods (KNN, K-means) are most sensitive to extreme values - a single "
            "high-leverage point can shift a regression line substantially. "
            f"{
                'At this rate, winsorising is preferable to dropping rows, as you'
                ' would lose too many observations.'
                if pct >= 0.10
                else 'Winsorising '
                '(capping at the IQR fence) or log-transforming are the standard '
                'mitigations.'
            } "
            "Tree-based models are largely robust to outliers in features but "
            "remain sensitive when they appear in the target variable."
        )

        ml_actions: list[dict[str, str]] = [
            {
                "action": "winsorize",
                "column": col,
                "detail": (
                    f"Cap values at IQR fence [{lower:.4g}, {upper:.4g}]"
                    if lower is not None and upper is not None
                    else "Cap at IQR fence"
                ),
            },
            {
                "action": "transform",
                "method": "log1p",
                "column": col,
                "condition": "right-skewed with positive values",
            },
            {
                "action": "investigate",
                "column": col,
                "detail": "Confirm whether extreme values are errors or genuine "
                "observations before transforming",
            },
        ]

        metric: dict[str, float | int | None] = {
            "outlier_count": count,
            "outlier_pct": round(pct, 4),
            "n_rows": n_rows,
            "iqr_lower": round(lower, 4) if lower is not None else None,
            "iqr_upper": round(upper, 4) if upper is not None else None,
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=f"{severity.title()} Outliers ({pct_str}, {count:,} rows)",
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )
        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title=f"Outlier Sensitivity ({pct_str} outside IQR fence)",
            body=ml_body.strip(),
            actions=ml_actions,
            metric=metric,
        )
