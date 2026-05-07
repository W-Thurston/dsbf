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
    phase="eda",
    tags=["redundancy", "skew"],
    expected_semantic_types=["any"],
)
class DetectSingleDominantValue(BaseTask):
    """
    Detects columns where one value dominates the distribution.

    For every column, computes the mode and its proportion. Columns where the
    mode proportion equals or exceeds ``dominance_threshold`` (default: 0.95)
    are counted as dominant.

    All columns are stored in ``data`` with their mode, proportion, unique value
    count, and dominance level - not just the flagged ones. This gives the
    data_quality_scorer a complete picture to work from.

    EDA and ML guidance blurbs are emitted for columns where mode proportion
    ≥ 0.70 (a lower threshold than the flag threshold, to surface informative
    findings before they become critical).

    Polars DataFrames are converted to pandas for the value_counts computation.

    Configurable parameters (via config["tasks"]["detect_single_dominant_value"]):
        dominance_threshold (float): Proportion above which a column is counted
            as having a dominant value. Default: 0.95
    """

    @staticmethod
    def _compute_dominance_level(proportion: float, unique_count: int) -> str:
        """
        Classify dominance relative to a uniform baseline.

        A proportion equal to 1/unique_count is the uniform baseline (no dominance).
        The dominance score measures how many times above that baseline the mode is.

        Args:
            proportion: Mode proportion (0.0 - 1.0).
            unique_count: Number of unique non-null values in the column.

        Returns:
            One of ``"very low"``, ``"low"``, ``"moderate"``, ``"high"``,
            or ``"very high"``.

        """
        if unique_count == 0:
            return "very low"

        expected_uniform: float = 1.0 / unique_count
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
        """
        Execute dominant value detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run_native("categorical")

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No categorical columns found — dominant value detection skipped.",
                    excluded,
                )
                return
            dominance_count = 0
            dominance_threshold = float(
                self.get_task_param("dominance_threshold") or 0.95,
            )

            if is_polars(df):
                # pandas value_counts used for proportion computation.
                df = df.to_pandas()

            results: dict[str, dict[str, Any]] = {}

            for col in df.columns:
                series = df[col].dropna()
                if series.empty:
                    continue

                value_counts = series.value_counts()
                top_val = value_counts.index[0]
                proportion = series.value_counts(normalize=True).iloc[0]
                unique_count: int = len(series.unique())

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
                        f"    '{col}' has dominant value '{top_val}' "
                        f"at {proportion:.1%}",
                        "debug",
                    )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Detected {dominance_count} column(s) with dominant "
                        f"values above {dominance_threshold:.0%}."
                    ),
                },
                data=results,
                metadata={
                    "dominance_threshold": dominance_threshold,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Redundancy",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            # Emit guidance at a lower threshold than the flag threshold so
            # findings surface before they become critical.
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
        """
        Generate EDA and ML guidance for a column with a dominant value.

        Args:
            col: Column name.
            col_data: Column stats dict containing ``mode``, ``mode_proportion``,
                and ``unique_values``.

        """
        mode = col_data["mode"]
        prop = col_data["mode_proportion"]
        unique = col_data["unique_values"]
        pct_str = f"{prop:.1%}"

        level: Literal["info", "warn"] = "warn" if prop >= 0.9 else "info"

        if prop >= 0.9:
            eda_body: str = (
                f"'{col}' is dominated by a single value: \"{mode}\" appears in "
                f"{pct_str} of rows. The column carries almost no variation - it is "
                f"close to constant. This could indicate a default value being "
                f"applied across most records, a data collection gap, or a genuine "
                f"characteristic of the population. Verify whether the rare "
                f'non-"{mode}" values are meaningful or noise.'
            )
            ml_body: str = (
                f"\"{mode}\" appears in {pct_str} of rows in '{col}'. Near-constant "
                f"features carry minimal predictive signal for any model and may "
                f"cause numerical instability in some algorithms. If used as a target "
                f"variable, the severe imbalance will bias predictions toward the "
                f"dominant class. Consider dropping, or apply class weighting and "
                f"stratified sampling if retained as a target."
            )
            ml_actions: list[dict] = [
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
                f"'{col}' has a dominant value: \"{mode}\" appears in {pct_str} of "
                f"rows across {unique} unique values. The distribution is uneven - "
                f"most observations share the same value while a minority are spread "
                f"across others. This is normal in many real-world categorical "
                f"columns, but worth noting when interpreting frequency counts or "
                f"group comparisons."
            )
            ml_body = (
                f"\"{mode}\" accounts for {pct_str} of '{col}'. If used as a "
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

        metric: dict = {
            "mode": mode,
            "mode_proportion": round(prop, 4),
            "unique_values": unique,
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=f'Dominant Value "{mode}" ({pct_str})',
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title=f'Class Imbalance - "{mode}" Dominates ({pct_str})',
            body=ml_body.strip(),
            actions=ml_actions,
            metric=metric,
        )
