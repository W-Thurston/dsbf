# dsbf/eda/tasks/compute_correlations.py

from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from pandas import DataFrame
from scipy.stats import chi2_contingency

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    log_reliability_warnings,
    make_failure_result,
)
from dsbf.utils.backend import is_polars


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Compute Cramér's V association statistic between two categorical Series.

    Cramér's V is a normalized measure of association derived from the chi-squared
    statistic. It ranges from 0 (no association) to 1 (perfect association) and
    is symmetric: cramers_v(x, y) == cramers_v(y, x).

    Args:
        x: First categorical Series.
        y: Second categorical Series. Must have the same index as x.

    Returns:
        Cramér's V value in [0.0, 1.0], or 0.0 if the contingency table
        is degenerate (fewer than 2 levels in either variable).

    """
    contingency: DataFrame = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    denom: int = min(k - 1, r - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


@register_task(
    display_name="Compute Correlations",
    description=(
        "Computes Pearson correlation for numeric pairs and Cramér's V "
        "for categorical pairs. Results are consumed by the Relationships tab "
        "and the dataset-level correlation heatmap."
    ),
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="moderate",
    phase="eda",
    tags=["numeric", "correlation", "categorical"],
    expected_semantic_types=["continuous", "categorical"],
)
class ComputeCorrelations(BaseTask):
    """
    Computes pairwise correlations across all numeric and categorical column pairs.

    Numeric pairs use Pearson correlation. Categorical pairs use Cramér's V.
    High-cardinality categorical columns are skipped to avoid OOM on large
    contingency tables.

    For Polars DataFrames, numeric correlation uses the native Polars `.corr()`
    matrix. Categorical correlation always converts to pandas since scipy's
    chi2_contingency requires numpy/pandas arrays.

    Reliability warnings are attached when data conditions (low N, outliers,
    high skew, zero variance) may distort Pearson results.

    Output is stored flat as ``{"col_a|col_b": value}`` and consumed by
    ``generate_dataset_summary_plots`` for heatmap rendering, and by the
    Relationships tab for pairwise association display.
    """

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """
        Execute correlation computation and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data
            correlations: dict[str, float] = {}
            backend_used: Literal["pandas", "polars"] = (
                "polars" if is_polars(df) else "pandas"
            )

            # High-cardinality categorical columns produce contingency tables that
            # can exhaust memory or hang for seconds per pair. Skip any column
            # exceeding this limit. Matches detect_high_cardinality default.
            cat_cardinality_limit = int(
                self.get_task_param("cat_cardinality_limit") or 50,
            )

            # --- Polars numeric correlation ---
            if is_polars(df):
                numeric_cols: list[str] = [
                    col for col in df.columns if df[col].dtype in (pl.Float64, pl.Int64)
                ]

                minimum_column_count = 2
                if len(numeric_cols) >= minimum_column_count:
                    try:
                        numeric_df = df.select(numeric_cols)
                        # Polars .corr() returns a square DataFrame where
                        # row i / column j is corr(col_i, col_j). Access via
                        # column name (column axis) and row index (row axis).
                        corr_matrix = numeric_df.corr()
                        for i, col1 in enumerate(numeric_cols):
                            for j in range(i + 1, len(numeric_cols)):
                                col2: str = numeric_cols[j]
                                value = corr_matrix[col2][i]
                                correlations[f"{col1}|{col2}"] = float(value)
                    except Exception as e:  # noqa: BLE001
                        self._log(
                            f"    Polars correlation failed: {e}."
                            " Falling back to pandas.",
                            "debug",
                        )
                        df = df.to_pandas()
                        backend_used = "pandas"
                else:
                    self._log(
                        "    Fewer than 2 numeric columns — skipping numeric "
                        "correlation.",
                        "debug",
                    )

            # --- Pandas numeric correlation ---
            if not is_polars(df):
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                minimum_column_count = 2
                if len(numeric_cols) < minimum_column_count:
                    self._log(
                        "    Fewer than 2 numeric columns — skipping numeric "
                        "correlation.",
                    )
                    self.output = TaskResult(
                        name=self.name,
                        status="success",
                        summary={
                            "message": (
                                "Not enough numeric columns to compute correlations."
                            ),
                        },
                        data={},
                    )
                    return

                for i, col1 in enumerate(numeric_cols):
                    for j in range(i + 1, len(numeric_cols)):
                        col2 = numeric_cols[j]
                        correlations[f"{col1}|{col2}"] = float(df[col1].corr(df[col2]))

            # --- Categorical Cramér's V ---
            # scipy requires pandas/numpy; convert Polars to pandas here if not
            # already done. This is the only forced conversion in this task.
            if is_polars(df):
                df = df.to_pandas()
                backend_used = "mixed"

            cat_cols = df.select_dtypes(include=["object"]).columns
            cat_unique: dict[str, int] = {
                col: int(df[col].nunique()) for col in cat_cols
            }
            skipped_high_card: list[str] = []

            for i, col1 in enumerate(cat_cols):
                for j in range(i + 1, len(cat_cols)):
                    col2 = cat_cols[j]
                    if (
                        cat_unique[col1] > cat_cardinality_limit
                        or cat_unique[col2] > cat_cardinality_limit
                    ):
                        skipped_high_card.append(f"{col1}|{col2}")
                        continue
                    correlations[f"{col1}|{col2}"] = cramers_v(df[col1], df[col2])

            skipped_msg: str = (
                f" ({len(skipped_high_card)} high-cardinality pairs skipped)."
                if skipped_high_card
                else "."
            )
            result = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed correlations for {len(correlations)} column pairs"
                        + skipped_msg
                    ),
                },
                data=correlations,
                metadata={
                    "backend": backend_used,
                    "cat_cardinality_limit": cat_cardinality_limit,
                    "skipped_high_cardinality_pairs": len(skipped_high_card),
                    "numeric_pair_count": sum(
                        1
                        for k in correlations
                        if "|" in k and k.split("|")[0] in numeric_cols
                    ),
                    "categorical_pair_count": sum(
                        1
                        for k in correlations
                        if "|" in k and k.split("|")[0] in cat_cols
                    ),
                    "suggested_viz_type": "heatmap",
                    "recommended_section": "Correlations",
                    "display_priority": "medium",
                    "column_types": self.get_column_type_info(list(df.columns)),
                },
            )

            # --- Reliability warnings ---
            flags: dict = self.ensure_reliability_flags()

            if flags["low_row_count"]:
                add_reliability_warning(
                    result,
                    level="strong_warning",
                    code="low_row_count",
                    description=(
                        "Pearson correlation may be statistically unreliable "
                        "with fewer than 30 observations."
                    ),
                    recommendation=(
                        "Use bootstrapped confidence intervals or collect more data."
                    ),
                )

            if flags["zero_variance_cols"]:
                add_reliability_warning(
                    result,
                    level="strong_warning",
                    code="zero_variance",
                    description=(
                        "The following features have near-zero variance: "
                        f"{flags['zero_variance_cols']}. Correlation is undefined."
                    ),
                    recommendation=(
                        "Drop or impute constant features before computing correlation."
                    ),
                )

            if flags["extreme_outliers"]:
                # Use a combined code when both outliers and low-N are present,
                # since the two conditions compound each other's unreliability.
                code: Literal = (
                    "extreme_outliers_low_n"
                    if flags["low_row_count"]
                    else "extreme_outliers"
                )
                description: Literal = (
                    "Some features contain extreme z-scores (|z| > 3), but sample "
                    "size is small (N < 30). Outlier estimates may be unreliable."
                    if flags["low_row_count"]
                    else (
                        "Some features contain extreme z-scores (|z| > 3), "
                        "which may distort Pearson correlation."
                    )
                )
                recommendation: Literal = (
                    "Interpret outlier influence with caution or validate using "
                    "robust statistics."
                    if flags["low_row_count"]
                    else "Winsorize outliers or use Spearman correlation."
                )
                add_reliability_warning(
                    result,
                    level="heuristic_caution",
                    code=code,
                    description=description,
                    recommendation=recommendation,
                )

            if flags["high_skew"]:
                code = "high_skew_low_n" if flags["low_row_count"] else "high_skew"
                description = (
                    "High skew was detected, but sample size is small (N < 30). "
                    "Skew estimates may be unstable."
                    if flags["low_row_count"]
                    else (
                        "One or more features are highly skewed, which may distort "
                        "correlation strength."
                    )
                )
                recommendation = (
                    "Interpret skewness cautiously or validate with bootstrapping."
                    if flags["low_row_count"]
                    else "Use Spearman correlation or log-transform skewed variables."
                )
                add_reliability_warning(
                    result,
                    level="heuristic_caution",
                    code=code,
                    description=description,
                    recommendation=recommendation,
                )

            log_reliability_warnings(self, result)
            self.output = result

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
