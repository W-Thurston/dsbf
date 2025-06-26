# dsbf/eda/tasks/detect_collinear_features.py

from typing import Any, Dict

import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def detect_collinear_features(df: Any, vif_threshold: float = 10.0) -> TaskResult:
    """
    Detects multicollinearity among numeric features using VIF.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.
        vif_threshold (float): VIF value above which features are flagged as collinear.

    Returns:
        TaskResult: VIF scores and flagged columns.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        numeric_df = df.select_dtypes(include=np.number).dropna()
        if numeric_df.shape[1] < 2:
            return TaskResult(
                name="detect_collinear_features",
                status="success",
                summary="Not enough numeric features to compute VIF.",
                data={"vif_scores": {}, "collinear_columns": []},
            )

        vif_scores: Dict[str, float] = {}
        for i in range(numeric_df.shape[1]):
            col = numeric_df.columns[i]
            vif_scores[col] = variance_inflation_factor(numeric_df.values, i)

        flagged = [col for col, vif in vif_scores.items() if vif > vif_threshold]

        return TaskResult(
            name="detect_collinear_features",
            status="success",
            summary=f"Flagged {len(flagged)} column(s) with VIF > {vif_threshold}.",
            data={"vif_scores": vif_scores, "collinear_columns": flagged},
            metadata={"vif_threshold": vif_threshold},
        )

    except Exception as e:
        return TaskResult(
            name="detect_collinear_features",
            status="failed",
            summary=f"Error during collinearity detection: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
