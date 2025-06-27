# dsbf/eda/tasks/detect_single_dominant_value.py

from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Single Dominant Value",
    description="Detects columns dominated by a single value.",
    depends_on=["infer_types"],
    stage="raw",
    tags=["redundancy", "skew"],
)
class DetectSingleDominantValue(BaseTask):
    """
    Detects columns where a single value dominates the distribution,
    such as binary features with a heavy skew or categorical columns
    where nearly all values are the same.
    """

    def run(self) -> None:
        try:
            df: Any = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            dominance_threshold: float = self.config.get("dominance_threshold", 0.95)
            results: Dict[str, Dict[str, Any]] = {}

            for col in df.columns:
                vc = df[col].value_counts(dropna=False, normalize=True)
                if not vc.empty and vc.iloc[0] >= dominance_threshold:
                    results[col] = {
                        "dominant_value": vc.index[0],
                        "proportion": float(vc.iloc[0]),
                    }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Detected {len(results)} column(s) with dominant values.",
                data=results,
                metadata={"dominance_threshold": dominance_threshold},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during dominant value detection: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
