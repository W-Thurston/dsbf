# dsbf/eda/tasks/detect_zeros.py

from typing import Any, Dict

import numpy as np

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Zeros",
    description="Flags columns or rows with high zero concentration.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="cleaned",
    tags=["zeros", "sparsity"],
)
class DetectZeros(BaseTask):
    """
    Detects numeric columns with a high proportion of zero values.
    Flags columns where zeros exceed a specified threshold.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            flag_threshold = float(self.get_task_param("flag_threshold") or 0.95)

            if is_polars(df):
                df = df.to_pandas()

            if not hasattr(df, "shape"):
                raise ValueError("Input is not a valid dataframe.")

            n_rows = df.shape[0]
            zero_counts: Dict[str, int] = {}
            zero_percentages: Dict[str, float] = {}
            zero_flags: Dict[str, bool] = {}

            numeric_df = df.select_dtypes(include=[np.number])

            for col in numeric_df.columns:
                count = int((numeric_df[col] == 0).sum())
                pct = count / n_rows
                zero_counts[col] = count
                zero_percentages[col] = pct
                zero_flags[col] = pct > flag_threshold

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Flagged {sum(zero_flags.values())}"
                        f"column(s) with high zero counts."
                    )
                },
                data={
                    "zero_counts": zero_counts,
                    "zero_percentages": zero_percentages,
                    "zero_flags": zero_flags,
                },
                metadata={
                    "threshold_pct": flag_threshold,
                    "total_rows": n_rows,
                },
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
