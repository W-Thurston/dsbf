# dsbf/eda/tasks/detect_single_dominant_value.py

from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Detect Single Dominant Value",
    description="Detects columns dominated by a single value.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
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

            # ctx = self.context
            df: Any = self.input_data

            dominance_threshold = float(
                self.get_task_param("dominance_threshold") or 0.95
            )

            if is_polars(df):
                df = df.to_pandas()

            results: Dict[str, Dict[str, Any]] = {}

            for col in df.columns:
                vc = df[col].value_counts(dropna=False, normalize=True)
                if not vc.empty and vc.iloc[0] >= dominance_threshold:
                    self._log(
                        f"{col} has dominant value {vc.index[0]} at {vc.iloc[0]:.1%}",
                        "debug",
                    )
                    results[col] = {
                        "dominant_value": vc.index[0],
                        "proportion": float(vc.iloc[0]),
                    }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Detected {len(results)} column(s) with dominant values."
                    )
                },
                data=results,
                metadata={"dominance_threshold": dominance_threshold},
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
