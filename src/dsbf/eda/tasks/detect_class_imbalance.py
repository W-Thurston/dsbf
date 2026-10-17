# dsbf/eda/tasks/detect_class_imbalance.py

import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    name="detect_class_imbalance",
    display_name="Detect Class Imbalance",
    description=(
        "Detects severe class imbalance in the configured target column "
        "and recommends mitigation strategies."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="modeling",
    domain="core",
    runtime_estimate="fast",
    phase="ml_readiness",
    tags=["target", "imbalance", "ml_readiness"],
    # Target column can be any semantic type - categorical targets are the
    # most common case but numeric binary targets (0/1) also apply.
    expected_semantic_types=["any"],
)
class DetectClassImbalance(BaseTask):
    """
    Detects class imbalance in a configured target column.

    Computes the class distribution and majority class ratio for the designated
    target column. When the majority class exceeds a configurable threshold
    (default: 0.9), the task emits both EDA and ML guidance blurbs and an
    ML impact signal.

    Supports both Polars and Pandas DataFrames.

    If no ``target_column`` is set in config, the task returns a skipped result.

    Configurable parameters (via config["tasks"]["detect_class_imbalance"]):
        target_column (str): Name of the target column to analyse.
        imbalance_ratio_threshold (float): Majority class proportion that triggers
            an imbalance flag. Default: 0.9
    """

    def run(self) -> None:
        """
        Execute class imbalance detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            target_col: str | None = self.get_task_param("target_column")
            threshold: float = float(
                self.get_task_param("imbalance_ratio_threshold") or 0.9,
            )

            if not target_col or target_col not in df.columns:
                self.output = TaskResult(
                    name=self.name,
                    status="skipped",
                    summary={
                        "message": (
                            "[SKIPPED] No valid target column configured "
                            "for imbalance detection."
                        ),
                    },
                    data={},
                    recommendations=[
                        "Set a valid `target_column` in config to enable "
                        "class imbalance checks.",
                    ],
                )
                return

            # Compute class distribution
            if is_polars(df):
                counts_df = (
                    df.group_by(target_col)
                    .agg(pl.len().alias("count"))
                    .sort("count", descending=True)
                )
                class_counts: dict = dict(
                    zip(
                        counts_df[target_col].to_list(),
                        counts_df["count"].to_list(),
                        strict=False,
                    ),
                )
            else:
                class_counts = dict(df[target_col].value_counts().to_dict())

            total: int = sum(class_counts.values())
            majority_class_count: int = (
                max(class_counts.values()) if class_counts else 0
            )
            majority_ratio: float = majority_class_count / total if total > 0 else 0.0
            is_imbalanced: bool = majority_ratio >= threshold

            recommendations: list[str] = []
            if is_imbalanced:
                recommendations.append(
                    "Class is highly imbalanced; consider upsampling, "
                    "downsampling, or reweighting.",
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Class imbalance analysis complete: majority class "
                        f"represents {majority_ratio:.2%} of total samples."
                    ),
                },
                data={
                    "target_column": target_col,
                    "class_distribution": class_counts,
                    "majority_ratio": round(majority_ratio, 4),
                    "imbalance_threshold": threshold,
                    "is_imbalanced": is_imbalanced,
                },
                recommendations=recommendations,
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Target",
                    "display_priority": "high",
                    "column_types": self.get_column_type_info([target_col]),
                },
            )

            if is_imbalanced:
                minority_ratio: float = 1.0 - majority_ratio
                majority_class = max(class_counts, key=class_counts.__getitem__)

                self.add_guidance(
                    result=self.output,
                    column=target_col,
                    phase="eda",
                    level="warn",
                    title=(
                        f"Class Imbalance - '{majority_class}' "
                        f"dominates ({majority_ratio:.1%})"
                    ),
                    body=(
                        f"'{target_col}' has a severely imbalanced class distribution: "
                        f"'{majority_class}' accounts for {majority_ratio:.1%} of all "
                        f"samples, leaving the minority class(es) at "
                        f"{minority_ratio:.1%}. "
                        f"This level of imbalance means a classifier that always "
                        f"predicts the majority class achieves {majority_ratio:.1%} "
                        f"accuracy without learning anything. Examine whether the "
                        f"imbalance reflects the true population distribution or an "
                        f"artefact of data collection, filtering, or labelling."
                    ),
                    actions=[],
                    metric={
                        "majority_ratio": round(majority_ratio, 4),
                        "majority_class": str(majority_class),
                        "threshold": threshold,
                        "n_classes": len(class_counts),
                    },
                )

                self.add_guidance(
                    result=self.output,
                    column=target_col,
                    phase="ml",
                    level="warn",
                    title="Severe Class Imbalance - Adjust Training Strategy",
                    body=(
                        f"'{target_col}' has {majority_ratio:.1%} majority class "
                        f"proportion. Standard accuracy will be misleading - "
                        f"use precision, recall, F1, or AUC-PR as primary metrics. "
                        f"For tree-based models, set class_weight='balanced' or use "
                        f"scale_pos_weight (XGBoost). For neural networks, use "
                        f"weighted cross-entropy loss. SMOTE or random oversampling "
                        f"of the minority class can be applied before training but "
                        f"must only be applied to the training split to avoid "
                        f"data leakage into validation."
                    ),
                    actions=[
                        {
                            "action": "set_class_weight",
                            "method": "class_weight='balanced'",
                            "column": target_col,
                            "detail": (
                                "Supported natively by sklearn estimators "
                                "and XGBoost scale_pos_weight"
                            ),
                        },
                        {
                            "action": "resample",
                            "method": "SMOTE or RandomOverSampler",
                            "column": target_col,
                            "condition": "apply only to training split",
                            "detail": "Oversample minority class before fitting",
                        },
                        {
                            "action": "change_metric",
                            "method": "AUC-PR or F1",
                            "column": target_col,
                            "detail": (
                                "Accuracy is uninformative under class imbalance"
                            ),
                        },
                    ],
                    metric={
                        "majority_ratio": round(majority_ratio, 4),
                        "majority_class": str(majority_class),
                        "threshold": threshold,
                    },
                )

            # ML impact scoring
            if self.get_engine_param("enable_impact_scoring", True) and is_imbalanced:
                tip: str | None = get_recommendation_tip(
                    self.name,
                    {"majority_ratio": majority_ratio},
                )
                self.set_ml_signals(
                    result=self.output,
                    score=0.8,
                    tags=["monitor", "resample"],
                    recommendation=tip
                    or (
                        f"Target column '{target_col}' is highly imbalanced "
                        f"({majority_ratio:.2%} majority class). Consider "
                        "resampling, reweighting, or using metrics like AUC/PR "
                        "instead of accuracy."
                    ),
                )
                self.output.summary["column"] = target_col

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
