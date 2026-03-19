# dsbf/eda/tasks/detect_outliers.py

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
    phase="eda",
    tags=["outliers", "numeric"],
    expected_semantic_types=["continuous"],
)
class DetectOutliers(BaseTask):
    """
    Detects numeric outliers using the IQR method.

    For each numeric column, computes Q1, Q3, IQR, and lower/upper fences
    (Q1 - 1.5xIQR and Q3 + 1.5xIQR). Values outside these fences are flagged
    as outliers.

    A column is included in ``outlier_flags`` as True when the proportion of
    outliers exceeds ``flag_threshold`` (default: 1%). EDA and ML guidance
    blurbs are attached for any column with at least one outlier.

    Supports both Polars and Pandas DataFrames — Polars input is converted to
    pandas before processing since the IQR fence computation uses pandas quantile.

    Note: ``method`` parameter is accepted but only IQR is currently implemented.
    Future methods (z-score, MAD) will be added under the same parameter key.

    Configurable parameters (via config["tasks"]["detect_outliers"]):
        method (str): Outlier detection method. Currently only ``"iqr"``
            is supported. Default: ``"iqr"``
        flag_threshold (float): Proportion of outliers above which a column
            is added to ``outlier_flags`` as True. Default: 0.01
    """

    def run(self) -> None:
        """
        Execute outlier detection and populate self.output.

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

            # method is read for future extensibility — only IQR is implemented.
            method = str(self.get_task_param("method") or "iqr")
            flag_threshold = float(self.get_task_param("flag_threshold") or 0.01)

            if is_polars(df):
                # pandas quantile is used for IQR fence computation.
                df = df.to_pandas()

            if not hasattr(df, "shape"):
                raise ValueError("Input is not a valid dataframe.")

            n_rows = df.shape[0]
            outlier_counts: dict[str, int] = {}
            outlier_flags: dict[str, bool] = {}
            outlier_rows: dict[str, list[int]] = {}
            iqr_bounds: dict[str, dict[str, float]] = {}

            numeric_df = df.select_dtypes(include=[np.number])

            for col in numeric_df.columns:
                series = numeric_df[col].dropna()

                if series.empty:
                    self._log(f"    '{col}' skipped: empty after dropna()", "debug")
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
                    "message": f"Detected outliers in {len(flagged_cols)} column(s).",
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
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            # Emit guidance for every column with at least one outlier.
            for col, count in outlier_counts.items():
                if count > 0:
                    pct = count / n_rows
                    self._attach_guidance(
                        col,
                        count,
                        pct,
                        n_rows,
                        iqr_bounds.get(col, {}),
                    )

            # ML impact scoring
            if self.get_engine_param("enable_impact_scoring", True) and flagged_cols:
                top_col: str = flagged_cols[0]
                n_outliers: int = outlier_counts[top_col]
                tip: str | None = get_recommendation_tip(
                    self.name,
                    {"n_outliers": n_outliers},
                )
                self.set_ml_signals(
                    result=self.output,
                    score=0.7,
                    tags=["monitor", "transform"],
                    recommendation=tip
                    or (
                        f"Column '{top_col}' contains {n_outliers} statistical "
                        "outliers. Consider log-transforming, winsorizing, or "
                        "using robust models."
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

    def _attach_guidance(
        self,
        col: str,
        count: int,
        pct: float,
        n_rows: int,
        bounds: dict[str, float],
    ) -> None:
        """
        Generate EDA and ML guidance for a column with IQR-detected outliers.

        Args:
            col: Column name.
            count: Number of outlier values.
            pct: Proportion of outlier values (count / n_rows).
            n_rows: Total row count in the dataset.
            bounds: IQR bounds dict with keys ``lower``, ``upper``, ``q1``,
                ``q3``, ``iqr``.

        """
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
            eda_tail = (
                "At this rate, calling them outliers is misleading — more than one "
                "in seven rows falls outside the normal range, which suggests the "
                "distribution is simply heavy-tailed or multimodal rather than "
                "contaminated by a few anomalous points."
            )
        elif pct >= 0.05:
            level = "warn"
            severity = "notable"
            eda_tail = (
                "This is a notable minority — enough to materially affect mean-based "
                "statistics but small enough that these could be genuine rare events."
            )
        else:
            level = "info"
            severity = "low-level"
            eda_tail = (
                "A small number of values sit unusually far from the bulk of the "
                "distribution."
            )

        eda_body: str = (
            f"'{col}' has {count:,} values ({pct_str} of {n_rows:,} rows) "
            f"{bounds_desc}. {eda_tail} Check whether these extreme values are "
            f"plausible for the domain, arise from measurement or entry errors, "
            f"or represent a distinct sub-population worth analysing separately."
        )

        winsorise_detail: str = (
            "At this rate, winsorising is preferable to dropping rows, as you would "
            "lose too many observations."
            if pct >= 0.10
            else "Winsorising (capping at the IQR fence) or log-transforming are the "
            "standard mitigations."
        )

        ml_body: str = (
            f"'{col}' has {count:,} outlier values ({pct_str}) by the IQR method. "
            f"Linear models (regression, SVM with RBF kernel) and distance-based "
            f"methods (KNN, K-means) are most sensitive to extreme values — a single "
            f"high-leverage point can shift a regression line substantially. "
            f"{winsorise_detail} Tree-based models are largely robust to outliers "
            f"in features but remain sensitive when they appear in the target variable."
        )

        ml_actions: list[dict] = [
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
                "detail": (
                    "Confirm whether extreme values are errors or genuine "
                    "observations before transforming"
                ),
            },
        ]

        metric: dict = {
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
