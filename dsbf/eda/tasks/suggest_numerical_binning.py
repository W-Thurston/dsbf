# dsbf/eda/tasks/suggest_numerical_binning.py

import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


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
            skew_threshold = float(self.get_task_param("skew_threshold") or 1.0)

            if is_polars(df):
                numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
            else:
                numeric_cols = list(df.select_dtypes(include="number").columns)

            suggestions = {}

            for col in numeric_cols:
                try:
                    if is_polars(df):
                        col_expr = pl.col(col).drop_nulls()
                        skew = df.select(col_expr.skew()).item()
                        std = df.select(col_expr.std()).item()
                        min_val = df.select(col_expr.min()).item()
                        max_val = df.select(col_expr.max()).item()
                    else:
                        series = df[col].dropna()
                        skew = float(series.skew())
                        std = float(series.std())
                        min_val = float(series.min())
                        max_val = float(series.max())

                    value_range = max_val - min_val

                    if (
                        skew is None
                        or pl.Series([skew]).is_nan().any()
                        or std is None
                        or std == 0
                        or value_range is None
                        or value_range == 0
                    ):
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

            summary = {
                "message": (
                    f"Binning suggestions generated for {len(suggestions)}"
                    " numeric columns."
                )
            }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=summary,
                data={"binning_suggestions": suggestions},
                recommendations=[
                    "Use quantile or equal-width binning for non-linear features. "
                    "Apply log transform to reduce high skew."
                ],
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
