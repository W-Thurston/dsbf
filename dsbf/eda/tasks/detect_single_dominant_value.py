# dsbf/eda/tasks/detect_single_dominant_value.py

from typing import Any, Literal

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
    expected_semantic_types=["any"],
)
class DetectSingleDominantValue(BaseTask):
    """
    Detects columns where a single value dominates the distribution,
    such as binary features with a heavy skew or categorical columns
    where nearly all values are the same.
    """

    @staticmethod
    def _compute_dominance_level(proportion: float, unique_count: int) -> str:
        """
        Assigns a dominance level based on how much the mode proportion
        exceeds a uniform distribution baseline.

        Returns:
            - A string label: low, moderate, high, very high
            - A dominance_score (float)

        """
        if unique_count == 0:
            return "very low"

        expected_uniform: float = 1 / unique_count
        dominance_score: float = (
            proportion / expected_uniform if expected_uniform > 0 else 0.0
        )

        if dominance_score < 1.2:
            return "very low"
        if dominance_score < 2:
            return "low"
        if dominance_score < 4:
            return "moderate"
        if dominance_score < 10:
            return "high"
        return "very high"

    def run(self) -> None:
        try:
            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_col)} column(s)", "debug")

            dominance_threshold = float(
                self.get_task_param("dominance_threshold") or 0.95,
            )
            dominance_count = 0

            if is_polars(df):
                df = df.to_pandas()

            results: dict[str, dict[str, Any]] = {}

            for col in df.columns:
                series = df[col].dropna()
                if series.empty:
                    continue  # Skip all-null columns

                value_counts = series.value_counts()

                top_val = value_counts.index[0]
                proportion = series.value_counts(normalize=True).iloc[0]
                unique_count: int = len(series.unique())

                # Always store mode information
                dominance_level: str = self._compute_dominance_level(
                    proportion,
                    unique_count,
                )

                results[col] = {
                    "mode": str(top_val),
                    "mode_proportion": round(proportion, 4),
                    "unique_values": unique_count,
                    "dominance_level": dominance_level,
                }

                if proportion >= dominance_threshold:
                    dominance_count += 1
                    self._log(
                        f"    {col} has dominant value {top_val} at {proportion:.1%}",
                        "debug",
                    )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Detected {dominance_count} column(s) with"
                        f" dominant values above {dominance_threshold:.0%}."
                    ),
                },
                data=results,
                plots={},
                metadata={
                    "dominance_threshold": dominance_threshold,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Redundancy",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys()),
                    ),
                },
            )

            # Generate guidance for columns with notable dominance (≥70%)
            guidance_threshold = 0.70
            for col, col_data in results.items():
                if col_data["mode_proportion"] >= guidance_threshold:
                    self._attach_guidance(col, col_data)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, col_data: dict[str, Any]) -> None:
        """Generate EDA + ML guidance for a column with a dominant value."""
        mode = col_data["mode"]
        prop = col_data["mode_proportion"]
        unique = col_data["unique_values"]
        pct_str: str = f"{prop:.1%}"

        level: Literal["info", "warn"] = "warn" if prop >= 0.9 else "info"

        if prop >= 0.9:
            eda_body: str = (
                f'{col} is dominated by a single value: "{mode}" appears in '
                f"{pct_str} of rows. The column carries almost no variation - it "
                f"is close to constant. This could indicate a default value being "
                f"applied across most records, a data collection gap, or a genuine "
                f"characteristic of the population. Verify whether the rare "
                f'non-"{mode}" values are meaningful or noise.'
            )
            ml_body: str = (
                f'"{mode}" appears in {pct_str} of rows in {col}. Near-constant '
                f"features provide minimal discriminative power to any model and may "
                f"cause numerical instability in some algorithms. If used as a target "
                f"variable, the severe imbalance will bias predictions toward the "
                f"dominant class. Consider dropping, or apply class weighting and "
                f"stratified sampling if retained as a target."
            )
            ml_actions: list[dict[str, str]] = [
                {
                    "action": "drop",
                    "column": col,
                    "detail": "Near-zero variance - minimal signal for any model",
                },
                {
                    "action": "class_weight",
                    "column": col,
                    "detail": "If used as classification target",
                },
                {
                    "action": "stratified_split",
                    "column": col,
                    "detail": "If used as classification target",
                },
            ]
        else:
            eda_body = (
                f'{col} has a dominant value: "{mode}" appears in {pct_str} of '
                f"rows across {unique} unique values. The distribution is uneven - "
                f"most observations share the same value while a minority are spread "
                f"across others. This is normal in many real-world categorical "
                f"columns, but worth noting when interpreting frequency counts or "
                f"group comparisons."
            )
            ml_body = (
                f'"{mode}" accounts for {pct_str} of {col}. If used as a '
                f"classification target, use stratified train/test splits to ensure "
                f"the minority classes are represented in both sets. Class weighting "
                f"may improve minority class recall."
            )
            ml_actions = [
                {
                    "action": "stratified_split",
                    "column": col,
                    "detail": "Ensure minority classes appear in train and test sets",
                },
                {
                    "action": "class_weight",
                    "column": col,
                    "detail": "Improve minority class recall if used as target",
                },
            ]

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=f'Dominant Value "{mode}" ({pct_str})',
            body=eda_body.strip(),
            actions=[],
            metric={
                "mode": mode,
                "mode_proportion": round(prop, 4),
                "unique_values": unique,
            },
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title=f'Class Imbalance - "{mode}" Dominates ({pct_str})',
            body=ml_body.strip(),
            actions=ml_actions,
            metric={
                "mode": mode,
                "mode_proportion": round(prop, 4),
                "unique_values": unique,
            },
        )
