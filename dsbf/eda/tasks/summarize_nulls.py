# dsbf/eda/tasks/summarize_nulls.py

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
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["nulls", "missing"],
    expected_semantic_types=["any"],
)
class SummarizeNulls(BaseTask):
    """
    Identify and summarize missing values across the dataset.

    Computes per-column null counts and percentages, identifies columns with
    high missingness, and analyses row-level null patterns (encoded as binary
    strings, e.g. ``"101"`` means columns 0 and 2 are null in that row).

    EDA and ML guidance blurbs are emitted for any column where the null
    percentage meets or exceeds 5%:

    - ≥ 50% → ``error`` level — imputation would introduce substantial bias
    - ≥ 20% → ``warn`` level — significant missingness requiring careful handling
    - ≥ 5%  → ``info`` level — manageable, standard imputation strategies apply

    Polars DataFrames are converted to pandas before processing.

    Configurable parameters (via config["tasks"]["summarize_nulls"]):
        null_threshold (float): Proportion above which a column is listed in
            ``high_null_columns``. Default: 0.5
    """

    def run(self) -> None:
        """
        Compute null statistics and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_cols)} column(s)", "debug")

            null_threshold = float(self.get_task_param("null_threshold") or 0.5)

            if is_polars(df):
                df = df.to_pandas()

            n_rows: int = df.shape[0]

            null_counts: dict[str, int] = df.isnull().sum().to_dict()
            null_percentages: dict[str, float] = {
                col: null_counts[col] / n_rows for col in df.columns
            }

            high_null_columns: list[str] = [
                col for col, pct in null_percentages.items() if pct >= null_threshold
            ]
            self._log(
                f"    Detected {len(high_null_columns)} columns with "
                f">{null_threshold:.0%} nulls",
                "debug",
            )

            # Row-level null pattern: "101" means column 0 and 2 are null in that row.
            null_patterns = (
                df.isnull()
                .astype(int)
                .apply(lambda row: "".join(row.astype(str)), axis=1)
            )
            pattern_counts: dict[str, int] = null_patterns.value_counts().to_dict()

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"{len(high_null_columns)} column(s) have "
                        f">{null_threshold:.0%} missing values."
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
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

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
        """
        Generate EDA and ML guidance for a column with notable missingness.

        Args:
            col: Column name.
            pct: Proportion of null values (0.0 - 1.0).
            count: Absolute count of null values.
            n_rows: Total row count in the dataset.

        """
        pct_str: str = f"{pct:.1%}"

        if pct >= 0.5:  # noqa: PLR2004
            level = "error"
            title: str = f"Severe Missingness ({pct_str})"
            eda_body: str = (
                f"'{col}' is missing {pct_str} of its values ({count:,} of "
                f"{n_rows:,} rows). More than half the data is absent — this column "
                f"is largely unobserved. Before drawing any conclusions, investigate "
                f"why so much data is missing: collection failure, a conditional "
                f"field only populated in certain cases, or a column that was not "
                f"available for most records?"
            )
            ml_body: str = (
                f"'{col}' has {pct_str} missing values. At this level of missingness "
                f"imputation will introduce substantial bias regardless of method. "
                f"Consider dropping the column unless the missingness itself is "
                f"informative — in which case retain a binary is_missing indicator "
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

        elif pct >= 0.2:  # noqa: PLR2004
            level = "warn"
            title = f"Significant Missingness ({pct_str})"
            eda_body = (
                f"'{col}' is missing {pct_str} of its values ({count:,} of "
                f"{n_rows:,} rows). This is substantial enough to affect any "
                f"analysis that uses this column. Consider whether the missing "
                f"values are random, or whether certain subgroups are more likely "
                f"to have data absent — a pattern in missingness can be as "
                f"informative as the values themselves."
            )
            ml_body = (
                f"'{col}' has {pct_str} missing values. Simple mean or mode "
                f"imputation will introduce bias at this level. Prefer median "
                f"imputation for skewed distributions, or model-based imputation "
                f"if data is likely missing not at random. Add a binary is_missing "
                f"indicator alongside imputed values to preserve the signal."
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
            level = "info"
            title = f"Some Missingness ({pct_str})"
            eda_body = (
                f"'{col}' is missing {pct_str} of its values ({count:,} of "
                f"{n_rows:,} rows). This is manageable but worth understanding — "
                f"check whether the missing rows share common characteristics "
                f"that might indicate a systematic gap rather than random absence."
            )
            ml_body = (
                f"'{col}' has {pct_str} missing values. Tree-based models handle "
                f"this natively in most frameworks. For linear models, impute "
                f"before fitting — mean or median imputation is reasonable at "
                f"this level. For time series, forward fill may be more appropriate."
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

        metric: dict[str, float | int] = {
            "null_pct": round(pct, 4),
            "null_count": count,
            "n_rows": n_rows,
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=title,
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title=title,
            body=ml_body.strip(),
            actions=ml_actions,
            metric=metric,
        )
