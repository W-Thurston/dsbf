# dsbf/eda/tasks/summarize_boolean_fields.py

from typing import Literal

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
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["boolean", "summary"],
    expected_semantic_types=["categorical"],
)
class SummarizeBooleanFields(BaseTask):
    """
    Summarize boolean and binary categorical columns.

    Identifies columns from the matched categorical set that have exactly 2
    unique non-null values (boolean-like). For each, computes the proportion
    of True, False, and null values.

    EDA and ML guidance blurbs are emitted for columns where the dominant
    value represents ≥ 75% of non-null rows, flagging potential class imbalance
    that would affect model training.

    Polars DataFrames are converted to pandas before the summary computation
    to allow consistent use of boolean comparison operators.
    """

    def run(self) -> None:
        """
        Execute boolean field summarization and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run_native()

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No boolean columns found — boolean field summary skipped.",
                    excluded,
                )
                return

            boolean_cols: list[str] = [
                col
                for col in matched_cols
                if (
                    df[col].drop_nulls().n_unique()
                    if is_polars(df)
                    else df[col].dropna().nunique()
                )
                == 2  # noqa: PLR2004
            ]
            # Categorical columns with > 2 values are excluded from this task.
            excluded.update(
                {
                    col: "categorical (>2 categories)"
                    for col in matched_cols
                    if col not in boolean_cols
                },
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
                summary={"message": f"Summarized {len(result)} boolean columns."},
                data=result,
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

            # Emit guidance for columns with notable imbalance (≥ 75% one class).
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
        dominant_val: bool,  # noqa: FBT001
        dominant_pct: float,
    ) -> None:
        """
        Generate EDA and ML guidance for an imbalanced boolean column.

        Args:
            col: Column name.
            stats: Dict with ``pct_true``, ``pct_false``, ``pct_null``.
            dominant_val: The dominant boolean value (True or False).
            dominant_pct: Proportion of non-null rows with the dominant value.

        """
        minority_pct: float = 1.0 - dominant_pct - stats["pct_null"]
        pct_str: str = f"{dominant_pct:.1%}"
        level: Literal["info", "warn"] = "warn" if dominant_pct >= 0.9 else "info"

        if dominant_pct >= 0.9:  # noqa: PLR2004
            eda_body: str = (
                f"'{col}' is severely imbalanced: {pct_str} of non-null rows are "
                f"{dominant_val}. The minority class ({minority_pct:.1%}) is rare "
                f"enough that meaningful patterns within it may be hard to observe. "
                f"Verify whether the rare class represents a genuine but uncommon "
                f"event, or whether it reflects a data collection gap or miscoding."
            )
            ml_body: str = (
                f"'{col}' has {pct_str} {dominant_val} values - severe class "
                f"imbalance. A naive classifier will achieve high accuracy by always "
                f"predicting {dominant_val}, while completely failing on the minority "
                f"class. Use stratified splits, class weighting, or oversampling "
                f"(SMOTE) if predicting this column. Evaluate with precision/recall "
                f"or F1, not accuracy."
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
                f"'{col}' is moderately imbalanced: {pct_str} of non-null rows are "
                f"{dominant_val}, with the remaining {minority_pct:.1%} being "
                f"{not dominant_val}. The minority class is present but "
                f"underrepresented. Check whether the split reflects the true "
                f"population or whether sampling introduced the imbalance."
            )
            ml_body = (
                f"'{col}' has moderate imbalance ({pct_str} {dominant_val}). "
                f"Use stratified train/test splits to ensure both classes are "
                f"represented proportionally. Class weighting is advisable if "
                f"minority class performance matters."
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

        metric: dict[str, float | str] = {
            "pct_true": round(stats["pct_true"], 4),
            "pct_false": round(stats["pct_false"], 4),
            "pct_null": round(stats["pct_null"], 4),
            "dominant_value": str(dominant_val),
            "dominant_pct": round(dominant_pct, 4),
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=(
                f"{'Severe' if dominant_pct >= 0.9 else 'Moderate'} "  # noqa: PLR2004
                f"Imbalance ({pct_str} {dominant_val})"
            ),
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title=f"Class Imbalance ({pct_str} {dominant_val})",
            body=ml_body.strip(),
            actions=ml_actions,
            metric=metric,
        )
