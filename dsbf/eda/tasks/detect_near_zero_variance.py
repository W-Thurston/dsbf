# dsbf/eda/tasks/detect_near_zero_variance.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    make_failure_result,
)
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    name="detect_near_zero_variance",
    display_name="Detect Near-Zero Variance",
    description="Flags numeric columns with extremely low variance.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="modeling",
    domain="core",
    runtime_estimate="fast",
    tags=["numeric", "variance", "ml_readiness"],
)
class DetectNearZeroVariance(BaseTask):
    def run(self) -> None:
        try:
            # df = self.input_data
            threshold = float(self.get_task_param("threshold") or 1e-4)

            flags = self.ensure_reliability_flags()
            low_variance = {
                col: round(var, 8)
                for col, var in flags["stds"].items()
                if var is not None and var**2 <= threshold
            }

            summary = {
                "message": f"{len(low_variance)} column(s) have near-zero variance."
            }

            recommendations = [
                "This column may add little modeling value — consider dropping."
                for _ in low_variance
            ]

            result = TaskResult(
                name=self.name,
                status="success",
                summary=summary,
                data={"low_variance_columns": low_variance},
                recommendations=recommendations,
            )

            if flags["zero_variance_cols"]:
                add_reliability_warning(
                    result,
                    level="strong_warning",
                    code="zero_variance",
                    description=(
                        "The following features have near-zero variance:"
                        f" {flags['zero_variance_cols']}."
                    ),
                    recommendation=(
                        "Drop or transform zero-variance" " features before modeling."
                    ),
                )

            self.output = result

            # Apply ML scoring to self.output
            if self.get_engine_param("enable_impact_scoring", True) and low_variance:
                col = next(iter(low_variance))
                var_val = low_variance[col]
                tip = get_recommendation_tip(self.name, {"variance": var_val})
                self.set_ml_signals(
                    result=result,
                    score=0.85,
                    tags=["drop"],
                    recommendation=tip
                    or (
                        f"Column '{col}' has near-zero variance (var = {var_val:.2e}). "
                        "Drop this feature to improve model"
                        " efficiency and reduce noise."
                    ),
                )
                result.summary["column"] = col

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
