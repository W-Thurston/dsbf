# dsbf/eda/tasks/detect_skewness.py

from typing import Any, Dict

import numpy as np
from scipy.stats import skew

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


class DetectSkewness(BaseTask):
    """
    Computes skewness for all numeric columns in a dataset.
    Skewness quantifies the asymmetry of a distribution.
    """

    def run(self) -> None:
        try:
            df: Any = self.input_data
            skewness: Dict[str, float] = {}

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
            else:
                # Use scipy's skew for Pandas
                numeric_df = df.select_dtypes(include="number")
                for col in numeric_df.columns:
                    skew_val = skew(numeric_df[col].dropna())
                    skewness[col] = float(skew_val)

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Computed skewness for {len(skewness)} numeric column(s).",
                data=skewness,
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during skewness detection: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
