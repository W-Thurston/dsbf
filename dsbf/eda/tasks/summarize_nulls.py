# dsbf/eda/tasks/summarize_nulls.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Summarize Nulls",
    description="Reports null value counts per column.",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["nulls", "missing"],
    expected_semantic_types=["any"],
)
class SummarizeNulls(BaseTask):
    """
    Identifies and summarizes missing values in a dataset.

    Computes:
    - Null counts per column
    - Null percentages per column
    - Columns with >50% missing values
    - Row-wise null patterns as binary strings (e.g., '101')
    """

    def run(self) -> None:
        try:
            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_col)} column(s)", "debug")

            null_threshold = float(self.get_task_param("null_threshold") or 0.5)

            if is_polars(df):
                df = df.to_pandas()

            n_rows: int = df.shape[0]

            # Column null counts and percentages
            null_counts: dict[str, int] = df.isnull().sum().to_dict()
            null_percentages: dict[str, float] = {
                col: null_counts[col] / n_rows for col in df.columns
            }

            high_null_columns: list[str] = [
                col for col, pct in null_percentages.items() if pct >= null_threshold
            ]
            self._log(
                f"    Detected {len(high_null_columns)} columns with >50% nulls",
                "debug",
            )

            # Row-wise null pattern frequency (e.g., "101" means null in cols 1 and 3)
            null_mask_df = df.isnull().astype(int)
            null_patterns = null_mask_df.apply(
                lambda row: "".join(row.astype(str)),
                axis=1,
            )
            pattern_counts: dict[str, int] = null_patterns.value_counts().to_dict()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"{len(high_null_columns)} column(s) have >50% missing values."
                    ),
                },
                data={
                    "null_counts": null_counts,
                    "null_percentages": null_percentages,
                    "high_null_columns": high_null_columns,
                    "null_patterns": pattern_counts,
                },
                metadata={
                    "rows": n_rows,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Missingness",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys()),
                    ),
                },
            )

            # Generate per-column guidance for any column with notable missingness
            guidance_threshold = 0.05
            for col, pct in null_percentages.items():
                if pct >= guidance_threshold:
                    self._attach_guidance(col, pct, null_counts[col], n_rows)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, pct: float, count: int, n_rows: int) -> None:
        """Generate EDA + ML guidance for a column with notable missingness."""
        pct_str: str = f"{pct:.1%}"

        if pct >= 0.5:
            level = "error"
            title: str = f"Severe Missingness ({pct_str})"
            eda_body: str = (
                f"{col} is missing {pct_str} of its values ({count:,} of {n_rows:,} "
                f"rows). More than half the data is absent - this column is largely "
                f"unobserved. Before drawing any conclusions from it, investigate why "
                f"so much data is missing: is this a collection failure, a conditional "
                f"field only populated in certain cases, or a column that simply was "
                f"not available for most records?"
            )
            ml_body: str = (
                f"{col} has {pct_str} missing values. At this level of missingness "
                f"imputation will introduce substantial bias regardless of method. "
                f"Consider dropping the column unless the missingness itself is "
                f"informative - in which case retain a binary is_missing indicator "
                f"and drop the original."
            )
            ml_actions: list[dict[str, str]] = [
                {
                    "action": "drop",
                    "column": col,
                    "detail": "Missingness too high to impute reliably",
                },
                {
                    "action": "add_indicator",
                    "method": "is_missing",
                    "column": col,
                    "detail": "If missingness pattern is informative",
                },
            ]

        elif pct >= 0.2:
            level = "warn"
            title = f"Significant Missingness ({pct_str})"
            eda_body = (
                f"{col} is missing {pct_str} of its values ({count:,} of {n_rows:,} "
                "rows). This is substantial enough to affect any analysis that uses "
                "this column. Consider whether the missing values are random, or "
                "whether certain subgroups are more likely to have data absent - "
                "a pattern in missingness can be as informative as the values "
                "themselves."
            )
            ml_body = (
                f"{col} has {pct_str} missing values. Simple mean or mode imputation "
                f"will introduce bias at this level. Prefer median imputation for "
                f"skewed distributions, or model-based imputation if data is likely "
                f"missing not at random. Add a binary is_missing indicator alongside "
                f"any imputed values to preserve the signal."
            )
            ml_actions = [
                {
                    "action": "impute",
                    "method": "median",
                    "column": col,
                    "condition": "skewed distribution",
                },
                {
                    "action": "impute",
                    "method": "model_based",
                    "column": col,
                    "condition": "missing not at random",
                },
                {
                    "action": "add_indicator",
                    "method": "is_missing",
                    "column": col,
                    "detail": "Retain missingness as a signal",
                },
            ]

        else:
            # 5-20%
            level = "info"
            title = f"Some Missingness ({pct_str})"
            eda_body = (
                f"{col} is missing {pct_str} of its values ({count:,} of {n_rows:,} "
                f"rows). This is manageable but worth understanding - check whether "
                f"the missing rows share any common characteristics that might "
                f"indicate a systematic gap rather than random absence."
            )
            ml_body = (
                f"{col} has {pct_str} missing values. Tree-based models handle this "
                f"natively in most frameworks. For linear models, impute before "
                f"fitting - mean or median imputation is reasonable at this level. "
                f"If time series, forward fill may be more appropriate."
            )
            ml_actions = [
                {
                    "action": "impute",
                    "method": "mean_or_median",
                    "column": col,
                    "condition": "linear models",
                },
                {
                    "action": "impute",
                    "method": "forward_fill",
                    "column": col,
                    "condition": "time series data",
                },
            ]

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=title,
            body=eda_body.strip(),
            actions=[],
            metric={"null_pct": round(pct, 4), "null_count": count, "n_rows": n_rows},
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title=title,
            body=ml_body.strip(),
            actions=ml_actions,
            metric={"null_pct": round(pct, 4), "null_count": count, "n_rows": n_rows},
        )
