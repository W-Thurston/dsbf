# dsbf/eda/tasks/detect_collinear_features.py

from typing import Dict, List

import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


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
        vif_threshold: float = self.config.get("vif_threshold", 10.0)

        try:
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            numeric_df = df.select_dtypes(include=np.number).dropna()

            # If not enough features, skip
            if numeric_df.shape[1] < 2:
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary="Not enough numeric features to compute VIF.",
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
                summary=(
                    f"Flagged {len(collinear_columns)} "
                    f"column(s) with VIF > {vif_threshold}."
                ),
                data={
                    "vif_scores": vif_scores,
                    "collinear_columns": collinear_columns,
                },
                metadata={"vif_threshold": vif_threshold},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during collinearity detection: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
