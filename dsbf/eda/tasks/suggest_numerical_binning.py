# dsbf/eda/tasks/suggest_numerical_binning.py

from typing import Literal

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    make_failure_result,
)
from dsbf.utils.backend import is_polars
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    name="suggest_numerical_binning",
    display_name="Suggest Numerical Binning",
    description=(
        "Suggests binning or log-transform strategies for numeric"
        " features with skewed or nonlinear distributions."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="modeling",
    domain="core",
    runtime_estimate="fast",
    tags=["numeric", "transformation", "ml_readiness"],
    expected_semantic_types=["continuous"],
)
class SuggestNumericalBinning(BaseTask):
    """
    Recommends binning or transformation methods for numeric features based on:
    - Skewness (log transform)
    - Distribution spread (equal-width vs quantile)
    """

    def run(self) -> None:
        try:
            df = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_col)} 'continuous' column(s)",
                "debug",
            )

            flags = self.ensure_reliability_flags()
            skew_vals = flags.get("skew_vals", {})
            stds = flags.get("stds", {})
            # means = flags.get("means", {})
            skew_threshold = float(self.get_task_param("skew_threshold") or 1.0)

            if is_polars(df):
                numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
            else:
                numeric_cols = list(df.select_dtypes(include="number").columns)

            suggestions = {}

            for col in numeric_cols:
                if col not in skew_vals or col not in stds or stds[col] == 0:
                    continue

                try:
                    skew = skew_vals[col]
                    std = stds[col]

                    if is_polars(df):
                        min_val = df[col].drop_nulls().min()
                        max_val = df[col].drop_nulls().max()
                    else:
                        min_val = df[col].dropna().min()
                        max_val = df[col].dropna().max()

                    value_range = max_val - min_val
                    if value_range is None or value_range == 0:
                        continue

                    if skew > skew_threshold:
                        strategy = "log-transform"
                    elif value_range > 3 * std:
                        strategy = "equal-width binning"
                    else:
                        strategy = "quantile binning"

                    suggestions[col] = {
                        "skewness": round(skew, 4),
                        "suggested_binning": strategy,
                    }

                except Exception:
                    continue

            result = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Binning suggestions generated for {len(suggestions)}"
                        " numeric columns."
                    ),
                },
                data={"binning_suggestions": suggestions},
                recommendations=[
                    "Use quantile or equal-width binning for non-linear features. "
                    "Apply log transform to reduce high skew.",
                ],
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Transformations",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys()),
                    ),
                },
            )

            for col, col_data in suggestions.items():
                self._attach_guidance(
                    col,
                    col_data["skewness"],
                    col_data["suggested_binning"],
                )

            if flags.get("low_row_count") and flags.get("high_skew"):
                add_reliability_warning(
                    result,
                    level="heuristic_caution",
                    code="binning_skew_low_n",
                    description=(
                        "Skewness-based binning strategies may be unstable with N < 30."
                    ),
                    recommendation=(
                        "Validate binning strategies with"
                        " visual plots or bootstrapping."
                    ),
                )

            self.output = result

            # Apply ML scoring to self.output
            if self.get_engine_param("enable_impact_scoring", True) and suggestions:
                col = next(iter(suggestions))
                strategy = suggestions[col]["suggested_binning"]
                skew_val = suggestions[col].get("skewness", 0.0)
                tip: str | None = get_recommendation_tip(
                    self.name,
                    {"strategy": strategy, "skew": skew_val},
                )
                self.set_ml_signals(
                    result=result,
                    score=0.6,
                    tags=["transform"],
                    recommendation=tip
                    or (
                        f"Column '{col}' shows distribution skew or spread. "
                        f"Recommended strategy: {strategy}."
                    ),
                )
                result.summary["column"] = col

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, skew: float, strategy: str) -> None:
        """
        Generate EDA + ML guidance for a numeric column's transformation posture.

        Strategy families (from the task's decision logic):
        - "log-transform"      - skew > skew_threshold (default 1.0)
        - "equal-width binning"- value_range > 3 * std (wide spread)
        - "quantile binning"   - otherwise (compact distribution)
        """
        skew_str: str = f"{skew:.4g}"
        abs_skew: float = abs(skew)
        direction: Literal["left", "right"] = "right" if skew > 0 else "left"

        if strategy == "log-transform":
            eda_level: Literal["info", "warn"] = "warn" if abs_skew > 2 else "info"
            eda_title: str = f"High Skew ({skew_str}) - Log Transform Suggested"
            eda_body: str = (
                f"{col} has a skewness of {skew_str} - a {direction}-skewed "
                f"distribution with a long {'upper' if skew > 0 else 'lower'} tail. "
                f"The mean is pulled {'above' if skew > 0 else 'below'} the median by "
                "the tail values. Summary statistics based on the mean (standard "
                "deviation, confidence intervals) will be distorted. Check the "
                "histogram to see whether the tail is driven by a few extreme values "
                "or by a smooth long-tailed distribution - the distinction matters for"
                " how you treat it."
            )
            ml_body: str = (
                f"{col} has skewness {skew_str}. A log transform (log1p for zero-safe "
                f"application) will compress the tail and make the distribution "
                f"approximately symmetric, which is beneficial for: linear models "
                f"(regression, SVM) that assume or prefer normality; distance-based "
                f"models (KNN, K-means) where large values dominate distances; and "
                f"neural networks where extreme inputs slow convergence. Tree-based "
                f"models are invariant to monotonic transformations - log transform "
                f"does not hurt them but also provides no benefit."
            )
            ml_actions: list[dict[str, str]] = [
                {
                    "action": "transform",
                    "method": "log1p",
                    "column": col,
                    "condition": "all values >= 0",
                },
                {
                    "action": "transform",
                    "method": "box_cox",
                    "column": col,
                    "condition": "all values > 0, needs scipy",
                },
                {
                    "action": "transform",
                    "method": "yeo_johnson",
                    "column": col,
                    "condition": "handles negative values",
                },
            ]

        elif strategy == "equal-width binning":
            eda_level = "info"
            eda_title = "Wide Spread - Equal-Width Binning Suggested"
            eda_body = (
                f"{col} has a wide value range relative to its spread "
                f"(skewness: {skew_str}). The distribution is roughly symmetric but "
                "spans a large absolute range. Equal-width bins (each covering the "
                "same value interval) will give an accurate picture of how values are "
                "distributed across the range. Verify that the range is not dominated "
                "by a small number of extreme values - if it is, quantile bins may "
                "give a more informative view."
            )
            ml_body = (
                f"{col} has a wide range with moderate skew ({skew_str}). "
                "Binning discretises the feature into ordinal categories, "
                "which can improve performance in tree models by creating explicit "
                "decision boundaries and reduce noise in linear models. Equal-width "
                "binning preserves the original scale's semantics. Choose the number "
                "of bins based on what splits are meaningful in the domain - "
                "typically 5-20. If the goal is normality rather than discretisation, "
                "standardise instead."
            )
            ml_actions = [
                {
                    "action": "bin",
                    "method": "equal_width",
                    "column": col,
                    "detail": "Use pd.cut() with uniform interval bins",
                },
                {
                    "action": "scale",
                    "method": "standard_scaler",
                    "column": col,
                    "condition": "if discretisation is not needed, standardise instead",
                },
            ]

        else:
            # quantile binning
            eda_level = "info"
            eda_title = "Compact Distribution - Quantile Binning Suggested"
            eda_body = (
                f"{col} has low skew ({skew_str}) and a compact spread relative to its"
                " range. Quantile bins will group values into categories of equal"
                " frequency, ensuring each bin has roughly the same number of "
                "observations. This is preferable to equal-width binning when the "
                "distribution is concentrated in a narrow range but the absolute "
                "values vary widely."
            )
            ml_body = (
                f"{col} has a compact, near-symmetric distribution "
                f"(skewness: {skew_str}). Quantile binning (pd.qcut) creates bins with"
                " equal row counts rather than equal value intervals - each category "
                "is equally represented in training, which can reduce class imbalance "
                "in the binned feature and improve model stability for tree models. "
                "For linear models, standardisation is typically preferable to binning"
                " for compact numeric features."
            )
            ml_actions = [
                {
                    "action": "bin",
                    "method": "quantile",
                    "column": col,
                    "detail": "Use pd.qcut() for equal-frequency bins",
                },
                {
                    "action": "scale",
                    "method": "standard_scaler",
                    "column": col,
                    "condition": "preferred over binning for linear models",
                },
            ]

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=eda_level,
            title=eda_title,
            body=eda_body.strip(),
            actions=[],
            metric={"skewness": round(skew, 4), "suggested_strategy": strategy},
        )
        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=eda_level,
            title=f"{strategy.title()} - ML Posture",
            body=ml_body.strip(),
            actions=ml_actions,
            metric={"skewness": round(skew, 4), "suggested_strategy": strategy},
        )
