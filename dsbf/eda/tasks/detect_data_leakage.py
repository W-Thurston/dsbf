# dsbf/eda/tasks/detect_data_leakage.py

from typing import Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Data Leakage",
    description="Heuristically detects columns that may leak target information.",
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="modeling",
    tags=["leakage", "target"],
)
class DetectDataLeakage(BaseTask):
    """
    Detects potential data leakage by identifying highly correlated numeric features.

    Flags any column pairs with absolute correlation >= threshold.
    """

    def run(self) -> None:
        """
        Run the data leakage detection task.

        Produces a TaskResult containing:
        - leakage_pairs: dict of "col1|col2" → float correlation
        """

        try:

            # ctx = self.context
            df = self.input_data

            correlation_threshold = float(
                self.get_task_param("correlation_threshold") or 0.99
            )

            if is_polars(df):
                self._log(
                    "Falling back to Pandas: correlation matrix requires numeric types",
                    "debug",
                )
                df = df.to_pandas()

            numeric_df = df.select_dtypes(include="number")
            corr_matrix = numeric_df.corr().abs()
            leakage_pairs: Dict[str, float] = {}

            # Scan upper triangle for highly correlated pairs
            for i, col1 in enumerate(corr_matrix.columns):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val >= correlation_threshold:
                        key = f"{col1}|{col2}"
                        leakage_pairs[key] = float(corr_val)

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Found {len(leakage_pairs)} highly correlated feature pairs."
                    )
                },
                data={"leakage_pairs": leakage_pairs},
                metadata={"correlation_threshold": correlation_threshold},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary={"message": (f"Error during data leakage detection: {e}")},
                data=None,
                metadata={"exception": type(e).__name__},
            )
