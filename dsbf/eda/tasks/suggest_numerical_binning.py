# dsbf/eda/tasks/suggest_numerical_binning.py

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

            matched_col, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_col)} 'continuous' column(s)", "debug"
            )

            flags = self.ensure_reliability_flags()
            skew_vals = flags.get("skew_vals", {})
            stds = flags.get("stds", {})
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
                    )
                },
                data={"binning_suggestions": suggestions},
                recommendations=[
                    "Use quantile or equal-width binning for non-linear features. "
                    "Apply log transform to reduce high skew."
                ],
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Transformations",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
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

            # CRITICAL: self.output must be assigned before any add_guidance calls.
            # add_guidance accesses self.output.guidance — if self.output is None
            # at that point, the task fails with 'NoneType has no attribute guidance'.
            self.output = result

            # ── Guidance blurbs ───────────────────────────────────────────────────
            for col, info in suggestions.items():
                skew = info["skewness"]
                strategy = info["suggested_binning"]

                # ── EDA blurb ────────────────────────────────────────────────────
                if strategy == "log-transform":
                    eda_level = "warn"
                    eda_title = "Right-skewed distribution — log transform suggested"
                    eda_body = (
                        f"'{col}' has a skewness of {skew:.2f}, indicating a "
                        f"right-skewed distribution where a few large values pull "
                        f"the tail. Skewed distributions compress most values into "
                        f"a narrow band, making patterns harder to see. Check the "
                        f"histogram for a long right tail and consider whether "
                        f"extreme values are real or anomalous."
                    )
                elif strategy == "equal-width binning":
                    eda_level = "info"
                    eda_title = "Wide value range — equal-width binning may help"
                    eda_body = (
                        f"'{col}' has a value range that exceeds 3× its standard "
                        f"deviation (skewness {skew:.2f}), suggesting values are "
                        f"spread across a wide scale. Equal-width binning can reveal "
                        f"where values cluster within that range. Inspect whether "
                        f"the spread reflects genuine variation or outlier influence."
                    )
                else:  # quantile binning
                    eda_level = "info"
                    eda_title = "Moderate spread — quantile binning suitable"
                    eda_body = (
                        f"'{col}' has a compact, roughly symmetric distribution "
                        f"(skewness {skew:.2f}) with spread close to its standard "
                        f"deviation. Quantile binning creates equal-frequency groups. "
                        f"Check the distribution for multimodality or unusual gaps "
                        f"before binning."
                    )

                self.add_guidance(
                    col=col,
                    phase="eda",
                    level=eda_level,
                    title=eda_title,
                    body=eda_body,
                    actions=[],
                    metric={"skewness": skew, "suggested_strategy": strategy},
                )

                # ── ML blurb ─────────────────────────────────────────────────────
                if strategy == "log-transform":
                    ml_body = (
                        f"'{col}' is right-skewed (skewness {skew:.2f}). Linear "
                        f"models, regularized regression (Ridge, Lasso), and "
                        f"distance-based models (KNN, SVM) are sensitive to scale "
                        f"and skew — applying log1p before training improves "
                        f"coefficient stability and distance metrics. Tree-based "
                        f"models (Random Forest, XGBoost) are scale-invariant and "
                        f"do not require this transformation."
                    )
                    ml_actions = [
                        {
                            "action": "Apply log1p transform",
                            "method": "np.log1p",
                            "column": col,
                            "condition": "all values >= 0",
                        },
                        {
                            "action": (
                                "Verify no zero or negative values before transform"
                            ),
                            "method": "assert (df[col] >= 0).all()",
                            "column": col,
                            "condition": "pre-transform check",
                        },
                    ]
                elif strategy == "equal-width binning":
                    ml_body = (
                        f"'{col}' spans a wide value range relative to its spread "
                        f"(skewness {skew:.2f}). Equal-width binning discretizes "
                        f"the range into fixed-size intervals, which can help linear "
                        f"models capture non-linear relationships. For tree-based "
                        f"models this is generally unnecessary. Choose bin count "
                        f"based on the number of distinct clusters visible in the "
                        f"distribution."
                    )
                    ml_actions = [
                        {
                            "action": "Apply equal-width binning",
                            "method": "pd.cut",
                            "column": col,
                            "condition": "choose n_bins based on distribution",
                        },
                    ]
                else:  # quantile
                    ml_body = (
                        f"'{col}' has a compact distribution (skewness {skew:.2f}) "
                        f"suitable for quantile binning. Quantile bins ensure equal "
                        f"sample counts per group, reducing the impact of outliers "
                        f"on bin boundaries. Tree-based models are invariant to "
                        f"this transformation."
                    )
                    ml_actions = [
                        {
                            "action": "Apply quantile binning",
                            "method": "pd.qcut",
                            "column": col,
                            "condition": "duplicates='drop' if duplicate edges occur",
                        },
                    ]

                self.add_guidance(
                    col=col,
                    phase="ml",
                    level="info",
                    title=f"Binning strategy: {strategy}",
                    body=ml_body,
                    actions=ml_actions,
                    metric={"skewness": skew, "suggested_strategy": strategy},
                )

            # ── ML impact scoring ─────────────────────────────────────────────────
            if self.get_engine_param("enable_impact_scoring", True) and suggestions:
                col = next(iter(suggestions))
                strategy = suggestions[col]["suggested_binning"]
                skew_val = suggestions[col].get("skewness", 0.0)
                tip = get_recommendation_tip(
                    self.name, {"strategy": strategy, "skew": skew_val}
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
