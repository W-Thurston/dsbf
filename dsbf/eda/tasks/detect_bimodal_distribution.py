# dsbf/eda/tasks/detect_bimodal_distribution.py

from typing import Dict

import numpy as np
from sklearn.mixture import GaussianMixture

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


class DetectBimodalDistribution(BaseTask):
    """
    Uses Gaussian Mixture Models to flag numeric columns likely to follow
    a bimodal distribution, based on BIC improvement between 1 and 2 components.
    """

    def run(self) -> None:
        """
        Runs the bimodal detection task using Gaussian Mixture Models.
        Stores results as a TaskResult, including:
        - bimodal_flags: dict of column: bool
        - bic_scores: dict of column: {bic_1_component, bic_2_components}
        """
        bic_threshold: float = self.config.get("bic_threshold", 10.0)
        bimodal_flags: Dict[str, bool] = {}
        bic_scores: Dict[str, Dict[str, float]] = {}

        try:
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()  # sklearn requires numpy/pandas

            numeric_df = df.select_dtypes(include=np.number)

            for col in numeric_df.columns:
                col_data = numeric_df[col].dropna().values.reshape(-1, 1)

                # Skip if not enough data points for GMM
                if col_data.shape[0] < 10:
                    continue

                try:
                    gmm1 = GaussianMixture(n_components=1).fit(col_data)
                    gmm2 = GaussianMixture(n_components=2).fit(col_data)
                    bic1 = gmm1.bic(col_data)
                    bic2 = gmm2.bic(col_data)

                    bic_scores[col] = {
                        "bic_1_component": float(bic1),
                        "bic_2_components": float(bic2),
                    }
                    bimodal_flags[col] = bool((bic1 - bic2) > bic_threshold)
                except Exception as e:
                    print(f"[DetectBimodalDistribution] Failed on column {col}: {e}")
                    continue

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=(
                    f"Flagged {sum(bimodal_flags.values())} "
                    f"column(s) as likely bimodal."
                ),
                data={
                    "bimodal_flags": bimodal_flags,
                    "bic_scores": bic_scores,
                },
                metadata={"bic_threshold": bic_threshold},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during bimodal distribution detection: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
