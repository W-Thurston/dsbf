# dsbf/eda/tasks/summarize_numeric.py

from typing import Any, Dict

import numpy as np

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


class SummarizeNumeric(BaseTask):
    """
    Produces extended summary statistics for all numeric columns.

    Statistics include:
    - Count, mean, std, min, max
    - Percentiles: 1%, 5%, 25%, 50%, 75%, 95%, 99%
    - A flag for near-zero variance columns (variance < 1e-4)
    """

    def run(self) -> None:
        try:
            df: Any = self.input_data

            if is_polars(df):
                df = df.to_pandas()

            numeric_df = df.select_dtypes(include=np.number)
            extended_stats: Dict[str, Dict[str, Any]] = {}

            for col in numeric_df.columns:
                series = numeric_df[col].dropna()

                # Compute descriptive stats with extended percentiles
                desc = series.describe(percentiles=[0.01, 0.05, 0.95, 0.99])

                # Custom variance check for near-constant features
                variance = np.var(series)
                near_zero_var = variance < 1e-4

                extended_stats[col] = {
                    "count": desc["count"],
                    "mean": desc["mean"],
                    "std": desc["std"],
                    "min": desc["min"],
                    "1%": desc["1%"],
                    "5%": desc["5%"],
                    "25%": desc["25%"],
                    "50%": desc["50%"],
                    "75%": desc["75%"],
                    "95%": desc["95%"],
                    "99%": desc["99%"],
                    "max": desc["max"],
                    "near_zero_variance": near_zero_var,
                }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Extended summary for {len(extended_stats)} numeric columns.",
                data=extended_stats,
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during numeric summarization: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
