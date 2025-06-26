# dsbf/eda/tasks/detect_single_dominant_value.py

from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


class DetectSingleDominantValue(BaseTask):
    """
    Detects columns where a single value dominates the distribution,
    such as binary features with a heavy skew or categorical columns
    where nearly all values are the same.
    """

    def __init__(self, dominance_threshold: float = 0.95):
        """
        Args:
            dominance_threshold (float): Proportion above which a single value
                is considered dominant (e.g., 0.95 means 95% or more).
        """
        super().__init__()
        self.dominance_threshold = dominance_threshold

    def run(self) -> None:
        try:
            df: Any = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            results: Dict[str, Dict[str, Any]] = {}

            for col in df.columns:
                vc = df[col].value_counts(dropna=False, normalize=True)
                if not vc.empty and vc.iloc[0] >= self.dominance_threshold:
                    results[col] = {
                        "dominant_value": vc.index[0],
                        "proportion": float(vc.iloc[0]),
                    }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Detected {len(results)} column(s) with dominant values.",
                data=results,
                metadata={"dominance_threshold": self.dominance_threshold},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during dominant value detection: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
