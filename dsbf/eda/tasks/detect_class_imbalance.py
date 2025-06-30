# dsbf/eda/tasks/detect_class_imbalance.py

from typing import Optional

import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    name="detect_class_imbalance",
    display_name="Detect Class Imbalance",
    description=(
        "Detects severe class imbalance in the target"
        " column using configurable threshold."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="modeling",
    tags=["target", "imbalance", "ml_readiness"],
)
class DetectClassImbalance(BaseTask):
    """
    Detects class imbalance in the target variable by computing the ratio of the
    majority class. If this ratio exceeds a configured threshold (default = 0.9),
    the task suggests mitigation strategies via the recommendations field.

    Supports both Pandas and Polars backends.
    """

    def run(self) -> None:
        try:
            df = self.input_data

            # Load parameters from config
            target_col: Optional[str] = self.get_task_param("target_column")
            threshold: float = float(
                self.get_task_param("imbalance_ratio_threshold") or 0.9
            )

            if not target_col or target_col not in df.columns:
                self.output = TaskResult(
                    name=self.name,
                    status="skipped",
                    summary={
                        "message": (
                            "[SKIPPED] No valid target column"
                            " configured for imbalance detection."
                        )
                    },
                    data={},
                    recommendations=[
                        (
                            "Set a valid `target_column` in config to"
                            " enable class imbalance checks."
                        )
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
                class_counts = dict(
                    zip(counts_df[target_col].to_list(), counts_df["count"].to_list())
                )
            else:
                class_counts = dict(df[target_col].value_counts().to_dict())

            total: int = sum(class_counts.values())
            majority_class_count: int = (
                max(class_counts.values()) if class_counts else 0
            )
            majority_ratio: float = majority_class_count / total if total > 0 else 0.0

            # Determine recommendations only; status is always 'success'
            recommendations = []
            is_imbalanced = False
            if majority_ratio >= threshold:
                is_imbalanced = True
                recommendations.append(
                    "Class is highly imbalanced; consider upsampling,"
                    " downsampling, or reweighting."
                )

            summary = {
                "message": (
                    f"Class imbalance analysis complete: Majority class represents "
                    f"{majority_ratio:.2%} of total samples."
                )
            }

            data = {
                "target_column": target_col,
                "class_distribution": class_counts,
                "majority_ratio": round(majority_ratio, 4),
                "imbalance_threshold": threshold,
                "is_imbalanced": is_imbalanced,
            }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=summary,
                data=data,
                recommendations=recommendations,
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
