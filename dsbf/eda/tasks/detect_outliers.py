# dsbf/eda/tasks/detect_outliers.py

from typing import Any, Dict, List

import numpy as np

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Outliers",
    description="Uses statistical heuristics to flag outlier values.",
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    tags=["outliers", "numeric"],
)
class DetectOutliers(BaseTask):
    """
    Detects numeric outliers using the IQR method. Flags columns exceeding
    a proportion threshold of outliers.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            method = str(self.get_task_param("method") or "iqr")
            flag_threshold = float(self.get_task_param("flag_threshold") or 0.01)

            if is_polars(df):
                df = df.to_pandas()

            if not hasattr(df, "shape"):
                raise ValueError("Input is not a valid dataframe.")

            n_rows = df.shape[0]
            outlier_counts: Dict[str, int] = {}
            outlier_flags: Dict[str, bool] = {}
            outlier_rows: Dict[str, List[int]] = {}

            numeric_df = df.select_dtypes(include=[np.number])

            for col in numeric_df.columns:
                series = numeric_df[col].dropna()
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (series < lower) | (series > upper)
                indices = series[mask].index.tolist()

                outlier_counts[col] = len(indices)
                outlier_rows[col] = indices
                outlier_flags[col] = len(indices) > flag_threshold * n_rows

            flagged_cols = [col for col, flagged in outlier_flags.items() if flagged]

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (f"Detected outliers in {len(flagged_cols)} column(s).")
                },
                data={
                    "outlier_counts": outlier_counts,
                    "outlier_flags": outlier_flags,
                    "outlier_rows": outlier_rows,
                },
                metadata={
                    "method": method,
                    "threshold_pct": flag_threshold,
                    "total_rows": n_rows,
                },
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary={"message": (f"Error during outlier detection: {e}")},
                data=None,
                metadata={"exception": type(e).__name__},
            )
