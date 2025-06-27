# dsbf/eda/tasks/detect_high_cardinality.py

from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task()
class DetectHighCardinality(BaseTask):
    """
    Detects columns with a number of unique values greater than a threshold.
    """

    def __init__(self, threshold: int = 50):
        super().__init__()
        self.threshold = threshold

    def run(self) -> None:
        """
        Execute the high-cardinality detection task and store the results in
            `self.output`.
        """
        try:
            df: Any = self.input_data
            results: Dict[str, int] = {}

            if is_polars(df):
                for col in df.columns:
                    try:
                        n_unique = df[col].n_unique()
                        if n_unique > self.threshold:
                            results[col] = n_unique
                    except Exception:
                        continue
            else:
                for col in df.columns:
                    try:
                        n_unique = df[col].nunique()
                        if n_unique > self.threshold:
                            results[col] = n_unique
                    except Exception:
                        continue

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Detected {len(results)} high-cardinality column(s).",
                data=results,
                metadata={"threshold": self.threshold},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during high-cardinality detection: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
