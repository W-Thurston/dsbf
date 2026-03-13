from typing import Any, Literal

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Summarize Boolean Fields",
    description="Summarizes frequency and distribution of boolean columns.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["boolean", "summary"],
    expected_semantic_types=["categorical"],
)
class SummarizeBooleanFields(BaseTask):
    """
    Summarizes boolean columns by computing proportions of True, False, and
    missing values.
    """

    def run(self) -> None:
        try:
            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            boolean_cols = [
                col for col in matched_col if df[col].dropna().nunique() == 2
            ]
            excluded.update(
                {
                    col: "categorical (>2 categories)"
                    for col in matched_col
                    if col not in boolean_cols
                }
            )

            self._log(
                f"    Processing {len(boolean_cols)} 'boolean' column(s)",
                "debug",
            )

            if is_polars(df):
                df = df.to_pandas()

            result: dict[str, dict[str, float]] = {}

            for col in boolean_cols:
                total: int = len(df[col])
                true_count = (df[col] == True).sum()  # noqa: E712
                false_count = (df[col] == False).sum()  # noqa: E712
                null_count = df[col].isnull().sum()

                result[col] = {
                    "pct_true": true_count / total,
                    "pct_false": false_count / total,
                    "pct_null": null_count / total,
                }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": (f"Summarized {len(result)} boolean columns.")},
                data=result,
                plots={},
                metadata={
                    "bool_columns": boolean_cols,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Summary",
                    "display_priority": "low",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        boolean_cols + list(excluded.keys()),
                    ),
                },
            )

            # Generate guidance for imbalanced boolean columns (≥75% one class)
            guidance_threshold = 0.75
            for col, stats in result.items():
                dominant_pct: float = max(stats["pct_true"], stats["pct_false"])
                if dominant_pct >= guidance_threshold:
                    dominant_val: bool = stats["pct_true"] >= stats["pct_false"]
                    self._attach_guidance(col, stats, dominant_val, dominant_pct)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(
        self,
        col: str,
        stats: dict[str, float],
        dominant_val: bool,
        dominant_pct: float,
    ) -> None:
        """Generate EDA + ML guidance for an imbalanced boolean column."""
        minority_pct: float = 1.0 - dominant_pct - stats["pct_null"]
        pct_str: str = f"{dominant_pct:.1%}"
        level: Literal["info", "warn"] = "warn" if dominant_pct >= 0.9 else "info"

        if dominant_pct >= 0.9:
            eda_body: str = (
                f"{col} is severely imbalanced: {pct_str} of non-null rows are "
                f"{dominant_val}. The minority class ({minority_pct:.1%}) is rare "
                f"enough that it may be difficult to observe meaningful patterns "
                f"within it. Verify whether the rare class represents a genuine "
                f"but uncommon event, or whether it reflects a data collection "
                f"gap or miscoding."
            )
            ml_body: str = (
                f"{col} has {pct_str} {dominant_val} values - severe class imbalance. "
                f"A naive classifier will achieve high accuracy by always predicting "
                f"{dominant_val}, while completely failing on the minority class. "
                f"Use stratified splits, class weighting, or oversampling (SMOTE) "
                f"if predicting this column. Evaluate with precision/recall or F1, "
                f"not accuracy."
            )
            ml_actions: list[dict[str, str]] = [
                {
                    "action": "stratified_split",
                    "column": col,
                    "detail": "Preserve class ratio in train/test splits",
                },
                {
                    "action": "class_weight",
                    "column": col,
                    "detail": "Set class_weight='balanced' in classifier",
                },
                {
                    "action": "oversample",
                    "method": "SMOTE",
                    "column": col,
                    "condition": "minority class too small to learn from",
                },
            ]
        else:
            eda_body = (
                f"{col} is moderately imbalanced: {pct_str} of non-null rows are "
                f"{dominant_val}, with the remaining {minority_pct:.1%} being "
                f"{not dominant_val}. The minority class is present but "
                "underrepresented. "
                "Check whether the split reflects the true population or whether "
                "sampling introduced the imbalance."
            )
            ml_body = (
                f"{col} has moderate imbalance ({pct_str} {dominant_val}). "
                "Use stratified train/test splits to ensure both classes are "
                "represented proportionally. Class weighting is advisable if "
                "minority class performance matters."
            )
            ml_actions = [
                {
                    "action": "stratified_split",
                    "column": col,
                    "detail": "Preserve class ratio in train/test splits",
                },
                {
                    "action": "class_weight",
                    "column": col,
                    "detail": "Set class_weight='balanced' if minority recall matters",
                },
            ]

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=(
                f"{'Severe' if dominant_pct >= 0.9 else 'Moderate'} "
                f"Imbalance ({pct_str} {dominant_val})"
            ),
            body=eda_body.strip(),
            actions=[],
            metric={
                "pct_true": round(stats["pct_true"], 4),
                "pct_false": round(stats["pct_false"], 4),
                "pct_null": round(stats["pct_null"], 4),
                "dominant_value": str(dominant_val),
                "dominant_pct": round(dominant_pct, 4),
            },
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title=f"Class Imbalance ({pct_str} {dominant_val})",
            body=ml_body.strip(),
            actions=ml_actions,
            metric={
                "pct_true": round(stats["pct_true"], 4),
                "pct_false": round(stats["pct_false"], 4),
                "dominant_value": str(dominant_val),
                "dominant_pct": round(dominant_pct, 4),
            },
        )
