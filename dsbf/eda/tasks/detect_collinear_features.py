# dsbf/eda/tasks/detect_collinear_features.py

from typing import Dict, List

import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
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
    """
    Detects multicollinearity among numeric features using
        Variance Inflation Factor (VIF).

    Flags features whose VIF score exceeds a configured threshold.
    """

    def run(self) -> None:
        """
        Run VIF-based multicollinearity detection on numeric columns.

        Produces a TaskResult with:
        - vif_scores: {column: float}
        - collinear_columns: [column names with VIF > threshold]
        """

        try:

            # ctx = self.context
            df = self.input_data

            vif_threshold = float(self.get_task_param("vif_threshold") or 10.0)

            if is_polars(df):
                self._log(
                    "Falling back to Pandas: VIF calculation requires NumPy arrays",
                    "debug",
                )
                df = df.to_pandas()

            numeric_df = df.select_dtypes(include=np.number).dropna()

            # If not enough features, skip
            if numeric_df.shape[1] < 2:
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary={
                        "message": ("Not enough numeric features to compute VIF.")
                    },
                    data={"vif_scores": {}, "collinear_columns": []},
                    metadata={"vif_threshold": vif_threshold},
                )
                return

            vif_scores: Dict[str, float] = {}
            for i in range(numeric_df.shape[1]):
                col = numeric_df.columns[i]
                vif_val = variance_inflation_factor(numeric_df.values, i)
                vif_scores[col] = float(vif_val)  # Cast to native float

            collinear_columns: List[str] = [
                col for col, vif in vif_scores.items() if vif > vif_threshold
            ]

            self.output = TaskResult(
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

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
