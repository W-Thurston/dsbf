# dsbf/eda/tasks/detect_id_columns.py

from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


class DetectIDColumns(BaseTask):
    """
    Detects columns likely to be unique identifiers (e.g., user IDs, UUIDs).
    A column is flagged if its number of unique values exceeds 95% of total rows.
    """

    def __init__(self, threshold_ratio: float = 0.95):
        super().__init__()
        self.threshold_ratio = threshold_ratio

    def run(self) -> None:
        try:
            df: Any = self.input_data
            n_rows = df.shape[0]
            threshold = self.threshold_ratio * n_rows
            results: Dict[str, str] = {}

            if is_polars(df):
                for col in df.columns:
                    try:
                        n_unique = df[col].n_unique()
                        if n_unique >= threshold:
                            results[col] = f"{n_unique} unique values (likely ID)"
                    except Exception:
                        continue
            else:
                for col in df.columns:
                    try:
                        n_unique = df[col].nunique()
                        if n_unique >= threshold:
                            results[col] = f"{n_unique} unique values (likely ID)"
                    except Exception:
                        continue

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Detected {len(results)} likely ID column(s).",
                data=results,
                metadata={"rows": n_rows, "threshold_ratio": self.threshold_ratio},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during ID column detection: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
