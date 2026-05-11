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
    phase="ml_readiness",
    tags=["numeric", "variance", "ml_readiness"],
    expected_semantic_types=["continuous"],
)
class DetectNearZeroVariance(BaseTask):
    """
    Flags numeric columns with variance at or below a configurable threshold.

    Near-zero variance features provide almost no predictive signal to any
    model. They slow training, inflate feature counts, and can cause numerical
    instability in gradient-based algorithms. This is the continuous analogue of
    ``detect_constant_columns`` - where constant columns have exactly zero
    variance, near-zero variance columns have nearly identical values with only
    tiny perturbations.

    Variance is computed from the precomputed reliability flags (standard
    deviations squared), so no additional DataFrame pass is needed.

    Configurable parameters (via config["tasks"]["detect_near_zero_variance"]):
        threshold (float): Maximum variance (std²) for a column to be flagged.
            Default: 1e-4
    """

    def run(self) -> None:
        """
        Execute near-zero variance detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'continuous' column(s)",
                "debug",
            )

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No continuous columns found — near-zero variance check skipped.",
                    excluded,
                )
                return

            threshold = float(self.get_task_param("threshold") or 1e-4)

            # Reliability flags include per-column standard deviations computed
            # in a single pass over the data - no additional DataFrame scan needed.
            flags: dict = self.ensure_reliability_flags()
            low_variance: dict[str, float] = {
                col: round(std**2, 8)
                for col, std in flags["stds"].items()
                if std is not None and std**2 <= threshold
            }

            result = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"{len(low_variance)} column(s) have near-zero variance."
                    ),
                },
                data={"low_variance_columns": low_variance},
                metadata={
                    "threshold": threshold,
                    "suggested_viz_type": "box",
                    "recommended_section": "Variance",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            if flags["zero_variance_cols"]:
                add_reliability_warning(
                    result,
                    level="strong_warning",
                    code="zero_variance",
                    description=(
                        "The following features have near-zero variance: "
                        f"{flags['zero_variance_cols']}."
                    ),
                    recommendation=(
                        "Drop or transform zero-variance features before modeling."
                    ),
                )

            # Assign output before guidance so _attach_guidance can write to it.
            self.output = result

            for col, var in low_variance.items():
                self._attach_guidance(col, var, threshold)

            # ML impact scoring
            if self.get_engine_param("enable_impact_scoring", True) and low_variance:
                top_col: str = next(iter(low_variance))
                var_val: float = low_variance[top_col]
                tip: str | None = get_recommendation_tip(
                    self.name,
                    {"variance": var_val},
                )
                self.set_ml_signals(
                    result=result,
                    score=0.85,
                    tags=["drop"],
                    recommendation=tip
                    or (
                        f"Column '{top_col}' has near-zero variance "
                        f"(var = {var_val:.2e}). Drop this feature to improve "
                        "model efficiency and reduce noise."
                    ),
                )
                result.summary["column"] = top_col

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, variance: float, threshold: float) -> None:
        """
        Generate EDA and ML guidance for a near-zero variance column.

        Args:
            col: Column name.
            variance: Computed variance value (std²).
            threshold: Configured near-zero variance threshold.

        """
        var_str: str = f"{variance:.2e}"

        eda_body: str = (
            f"'{col}' has a variance of {var_str}, which is at or below the "
            f"near-zero threshold of {threshold:.2e}. This means almost all values "
            f"in this column are identical or nearly identical - it is essentially "
            f"a constant feature with minor noise. Verify whether this reflects a "
            f"genuine property of the data or a data collection artefact (e.g. a "
            f"sensor stuck at a fixed reading)."
        )

        ml_body: str = (
            f"'{col}' has variance {var_str} - effectively constant. Features with "
            f"near-zero variance provide negligible predictive signal to any "
            f"model. In gradient-based models they can cause numerical instability. "
            f"In tree-based models they waste a split candidate slot at every node. "
            f"Drop this column before training."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="warn",
            title=f"Near-Zero Variance (var = {var_str})",
            body=eda_body.strip(),
            actions=[],
            metric={"variance": variance, "threshold": threshold},
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="warn",
            title="Near-Zero Variance - Drop Before Modeling",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "drop",
                    "column": col,
                    "detail": "Near-constant features provide no signal and risk "
                    "numerical instability in gradient-based models",
                },
            ],
            metric={"variance": variance, "threshold": threshold},
        )
