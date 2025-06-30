# dsbf/eda/tasks/detect_near_zero_variance.py

import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    name="detect_near_zero_variance",
    display_name="Detect Near-Zero Variance",
    description="Flags numeric columns with extremely low variance.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="modeling",
    tags=["numeric", "variance", "ml_readiness"],
)
class DetectNearZeroVariance(BaseTask):
    """
    Flags numeric columns where variance is below a configured threshold
    (default 1e-4), suggesting they may be uninformative for modeling.
    """

    def run(self) -> None:
        try:
            df = self.input_data
            threshold = float(self.get_task_param("threshold") or 1e-4)

            if is_polars(df):
                numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
            else:
                numeric_cols = list(df.select_dtypes(include="number").columns)

            low_variance = {}

            for col in numeric_cols:
                try:
                    if is_polars(df):
                        var = df.select(pl.col(col).drop_nulls().var()).item()
                    else:
                        var = float(df[col].dropna().var())

                    if var is None or pl.Series([var]).is_nan().any():
                        continue

                    if var <= threshold:
                        low_variance[col] = round(var, 8)

                except Exception:
                    continue

            summary = {
                "message": f"{len(low_variance)} column(s) have near-zero variance."
            }

            recommendations = [
                "This column may add little modeling value — consider dropping."
                for _ in low_variance
            ]

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=summary,
                data={"low_variance_columns": low_variance},
                recommendations=recommendations,
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
