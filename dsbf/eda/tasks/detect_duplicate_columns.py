# dsbf/eda/tasks/detect_duplicate_columns.py

from typing import List, Tuple

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


class DetectDuplicateColumns(BaseTask):
    """
    Detects columns that are exact duplicates of one another.
    Compares all unique column pairs using .equals().
    """

    def run(self) -> None:
        """
        Run duplicate column detection logic.
        Produces a TaskResult with a list of (col1, col2) tuples for identical columns.
        """
        try:
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            duplicate_pairs: List[Tuple[str, str]] = []
            columns = df.columns.tolist()
            seen = set()

            for i, col1 in enumerate(columns):
                for j in range(i + 1, len(columns)):
                    col2 = columns[j]
                    if (col1, col2) not in seen:
                        try:
                            if df[col1].equals(df[col2]):
                                duplicate_pairs.append((col1, col2))
                                seen.add((col1, col2))
                        except Exception as e:
                            print(
                                f"[DetectDuplicateColumns] Comparison failed for "
                                f"{col1} and {col2}: {e}"
                            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Found {len(duplicate_pairs)} duplicate column pair(s).",
                data={"duplicate_column_pairs": duplicate_pairs},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during duplicate column detection: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
