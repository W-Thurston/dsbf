# dsbf/eda/tasks/detect_near_zero_variance.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    make_failure_result,
)
from dsbf.utils.plot_factory import PlotFactory
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
    expected_semantic_types=["continuous"],
)
class DetectNearZeroVariance(BaseTask):
    def run(self) -> None:
        try:
            # df = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(matched_col)} 'continuous' column(s)", "debug")

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

            plots: dict[str, dict[str, Any]] = {}

            # Only run if context + output_dir + input is set
            if self.context and self.context.output_dir and self.input_data is not None:
                df = self.input_data

                if hasattr(df, "to_pandas"):  # Polars support
                    df = df.to_pandas()

                for col in low_variance:
                    if col not in df.columns:
                        continue
                    series = df[col].dropna()

                    save_path = self.get_output_path(f"{col}_boxplot.png")
                    static = PlotFactory.plot_boxplot_static(series, save_path)
                    interactive = PlotFactory.plot_boxplot_interactive(
                        series, annotations=[f"Variance = {low_variance[col]:.8f}"]
                    )

                    plots[col] = {
                        "static": static["path"],
                        "interactive": interactive,
                    }

            result = TaskResult(
                name=self.name,
                status="success",
                summary=summary,
                data={"low_variance_columns": low_variance},
                recommendations=recommendations,
                plots=plots,
                metadata={
                    "suggested_viz_type": "box",
                    "recommended_section": "Variance",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
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
