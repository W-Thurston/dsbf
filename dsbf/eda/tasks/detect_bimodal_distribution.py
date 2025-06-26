# dsbf/eda/tasks/detect_bimodal_distribution.py

from typing import Any, Dict

import numpy as np
from sklearn.mixture import GaussianMixture

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def detect_bimodal_distribution(df: Any, bic_threshold: float = 10.0) -> TaskResult:
    """
    Detects numeric columns likely to have a bimodal distribution using
        Gaussian Mixture Models.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.
        bic_threshold (float): Minimum BIC improvement between 1 and 2 components
            to flag as bimodal.

    Returns:
        TaskResult: Columns flagged as bimodal with BIC scores.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        numeric_df = df.select_dtypes(include=np.number)
        bimodal_flags: Dict[str, bool] = {}
        bic_scores: Dict[str, Dict[str, float]] = {}

        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna().values.reshape(-1, 1)
            if col_data.shape[0] < 10:
                continue

            gmm1 = GaussianMixture(n_components=1).fit(col_data)
            gmm2 = GaussianMixture(n_components=2).fit(col_data)
            bic1 = gmm1.bic(col_data)
            bic2 = gmm2.bic(col_data)

            bic_scores[col] = {
                "bic_1_component": float(bic1),
                "bic_2_components": float(bic2),
            }
            bimodal_flags[col] = bool((bic1 - bic2) > bic_threshold)

        return TaskResult(
            name="detect_bimodal_distribution",
            status="success",
            summary=(
                f"Flagged {sum(bimodal_flags.values())}" f"column(s) as likely bimodal."
            ),
            data={"bimodal_flags": bimodal_flags, "bic_scores": bic_scores},
            metadata={"bic_threshold": bic_threshold},
        )

    except Exception as e:
        return TaskResult(
            name="detect_bimodal_distribution",
            status="failed",
            summary=f"Error during bimodal distribution detection: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
