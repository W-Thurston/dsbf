# dsbf/eda/tasks/detect_collinear_features.py

from typing import Dict, List

import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    make_failure_result,
)
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Collinear Features",
    description="Detects highly collinear features that may cause multicollinearity.",
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="modeling",
    tags=["multicollinearity", "numeric"],
)
class DetectCollinearFeatures(BaseTask):
    def run(self) -> None:
        try:
            df = self.input_data
            flags = self.ensure_reliability_flags()

            vif_threshold = float(self.get_task_param("vif_threshold") or 10.0)

            if is_polars(df):
                self._log(
                    "Falling back to Pandas: VIF calculation requires NumPy arrays",
                    "debug",
                )
                df = df.to_pandas()

            numeric_df = df.select_dtypes(include=np.number).dropna()

            if numeric_df.shape[1] < 2:
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary={"message": "Not enough numeric features to compute VIF."},
                    data={"vif_scores": {}, "collinear_columns": []},
                    metadata={"vif_threshold": vif_threshold},
                )
                return

            vif_scores: Dict[str, float] = {}
            for i in range(numeric_df.shape[1]):
                col = numeric_df.columns[i]
                vif_val = variance_inflation_factor(numeric_df.values, i)
                vif_scores[col] = float(vif_val)

            collinear_columns: List[str] = [
                col for col, vif in vif_scores.items() if vif > vif_threshold
            ]

            result = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Flagged {len(collinear_columns)} "
                        f"column(s) with VIF > {vif_threshold}."
                    )
                },
                data={
                    "vif_scores": vif_scores,
                    "collinear_columns": collinear_columns,
                },
                metadata={"vif_threshold": vif_threshold},
            )

            # Reliability warnings
            if flags["low_row_count"]:
                add_reliability_warning(
                    result,
                    level="heuristic_caution",
                    code="vif_low_n",
                    description=(
                        "VIF values may be unstable when"
                        " sample size is small (N < 30)."
                    ),
                    recommendation=(
                        "Consider bootstrapping or collecting"
                        " more data before interpreting VIF."
                    ),
                )
            if flags["zero_variance_cols"]:
                add_reliability_warning(
                    result,
                    level="strong_warning",
                    code="vif_zero_variance",
                    description=(
                        "Some features have near-zero variance,"
                        " which can distort VIF calculations."
                    ),
                    recommendation=(
                        "Drop or transform zero-variance features"
                        " before running VIF."
                    ),
                )

            self.output = result

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
