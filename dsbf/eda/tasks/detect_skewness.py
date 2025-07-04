# dsbf/eda/tasks/detect_skewness.py

from typing import Any, Dict

import numpy as np
from scipy.stats import skew

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    display_name="Detect Skewness",
    description="Computes skewness of numeric columns.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["distribution", "skew"],
)
class DetectSkewness(BaseTask):
    """
    Computes skewness for all numeric columns in a dataset.
    Skewness quantifies the asymmetry of a distribution.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data
            skewness: Dict[str, float] = {}

            # Compute skewness
            if is_polars(df):
                # Use manual skewness computation for Polars
                for col in df.columns:
                    if df[col].dtype in ("i64", "f64"):
                        series = df[col].to_numpy()
                        mean = np.mean(series)
                        std = np.std(series)
                        if std != 0:
                            skew_val = float(np.mean(((series - mean) / std) ** 3))
                            skewness[col] = skew_val
                            self._log(f"{col} skewness: {skew_val:.4f}", "debug")
            else:
                # Use scipy's skew for Pandas
                numeric_df = df.select_dtypes(include="number")
                for col in numeric_df.columns:
                    skew_val = skew(numeric_df[col].dropna())
                    skewness[col] = float(skew_val)
                    self._log(f"{col} skewness: {skew_val:.4f}", "debug")

            # Build TaskResult
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed skewness for {len(skewness)} numeric column(s)."
                    )
                },
                data=skewness,
            )

            # Apply ML scoring to self.output
            if self.get_engine_param("enable_impact_scoring", True):
                for col, skew_val in skewness.items():
                    abs_skew = abs(skew_val)
                    if abs_skew <= 1:
                        continue  # No ML risk
                    score = 0.6 if abs_skew <= 2 else 0.8
                    result = self.output
                    if result:
                        tip = get_recommendation_tip(self.name, {"skew": skew_val})
                        self.set_ml_signals(
                            result=result,
                            score=score,
                            tags=["transform"],
                            recommendation=tip
                            or (
                                f"Column '{col}' has high skew "
                                f"(skew = {skew_val:.2f}). Consider transforming it."
                            ),
                        )
                        result.summary["column"] = col
                        break

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
