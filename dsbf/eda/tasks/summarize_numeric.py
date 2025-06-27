# dsbf/eda/tasks/summarize_numeric.py

from typing import Any, Dict

import numpy as np

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task()
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
                desc = series.describe(
                    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
                )

                # Custom variance check for near-constant features
                variance = np.var(series)
                near_zero_var = bool(variance < 1e-4)

                extended_stats[col] = {
                    "count": desc.get("count", np.nan),
                    "mean": desc.get("mean", np.nan),
                    "std": desc.get("std", np.nan),
                    "min": desc.get("min", np.nan),
                    "1%": desc.get("1%", np.nan),
                    "5%": desc.get("5%", np.nan),
                    "25%": desc.get("25%", np.nan),
                    "50%": desc.get("50%", np.nan),
                    "75%": desc.get("75%", np.nan),
                    "95%": desc.get("95%", np.nan),
                    "99%": desc.get("99%", np.nan),
                    "max": desc.get("max", np.nan),
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
