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
                    )
                },
                data={"binning_suggestions": suggestions},
                recommendations=[
                    "Use quantile or equal-width binning for non-linear features. "
                    "Apply log transform to reduce high skew."
                ],
            )

            if flags.get("low_row_count") and flags.get("high_skew"):
                add_reliability_warning(
                    result,
                    level="heuristic_caution",
                    code="binning_skew_low_n",
                    description=(
                        "Skewness-based binning strategies"
                        " may be unstable with N < 30."
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
            self.output = make_failure_result(self.name, e)
