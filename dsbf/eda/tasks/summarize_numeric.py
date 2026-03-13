# dsbf/eda/tasks/summarize_numeric.py

from typing import Any, Literal

import numpy as np

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Summarize Numeric Columns",
    description="Computes basic stats (mean, std, min, max, etc.) for numeric columns",
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["numeric", "summary"],
    expected_semantic_types=["continuous"],
)
class SummarizeNumeric(BaseTask):
    """
    Produces extended summary statistics for all numeric columns.

    Statistics include:
    - Count, mean, std, min, max
    - Percentiles: 1%, 5%, 25%, 50%, 75%, 95%, 99%
    - A flag for near-zero variance columns (variance < 1e-4)
    """

    def run(self) -> None:
        try:
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_col)} 'continuous' column(s)",
                "debug",
            )

            if is_polars(df):
                df = df.to_pandas()
                self._log(
                    "    Converting Polars to Pandas for numeric summarization",
                    "debug",
                )

            numeric_df = df.select_dtypes(include=np.number)
            extended_stats: dict[str, dict[str, Any]] = {}

            for col in numeric_df.columns:
                series = numeric_df[col].dropna()

                if series.empty:
                    self._log(f"    {col} skipped: empty after dropna()", "debug")
                    continue

                # Compute descriptive stats with extended percentiles
                desc = series.describe(
                    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                )
                # Custom variance check for near-constant features
                variance = np.var(series)
                near_zero_var = bool(variance < 1e-4)

                extended_stats[col] = {
                    "count": desc.get("count", np.nan),
                    "mean": desc.get("mean", np.nan),
                    "std": desc.get("std", np.nan),
                    "min": desc.get("min", np.nan),
                    "1%": desc.get("1%", np.nan),
                    "5%": desc.get("5%", np.nan),
                    "25%": desc.get("25%", np.nan),
                    "50%": desc.get("50%", np.nan),
                    "75%": desc.get("75%", np.nan),
                    "95%": desc.get("95%", np.nan),
                    "99%": desc.get("99%", np.nan),
                    "max": desc.get("max", np.nan),
                    "near_zero_variance": near_zero_var,
                }

            self._log(f"    Summarized {len(extended_stats)} numeric columns", "debug")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Extended summary for {len(extended_stats)} numeric columns."
                    ),
                },
                data=extended_stats,
                plots={},
                metadata={
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Summary",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
            )

            for col, stats in extended_stats.items():
                self._attach_guidance(col, stats)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, stats: dict[str, Any]) -> None:
        """
        Generate EDA + ML guidance for a numeric column.

        Triggers:
        - near_zero_variance: the column is effectively constant
        - mean/median gap > 0.5 std: distribution is asymmetric (complementary
          to detect_skewness - that task reports the skewness coefficient;
          this one reports the raw central tendency gap in interpretable units)
        """
        mean: Any | None = stats.get("mean")
        std: Any | None = stats.get("std")
        median: Any | None = stats.get("50%")
        near_zero_var = stats.get("near_zero_variance", False)

        # --- Near-zero variance ---
        if near_zero_var:
            self.add_guidance(
                result=self.output,
                column=col,
                phase="eda",
                level="warn",
                title="Near-Zero Variance",
                body=(
                    f"{col} has a variance close to zero - nearly all values are "
                    f"identical (mean: {mean:.4g}, std: {std:.4g}). This column "
                    f"carries almost no variation across rows. Check whether it "
                    f"is a constant default, a derived field that only changes "
                    f"under rare conditions, or a data loading artifact."
                ),
                actions=[],
                metric={
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "variance": round(std**2, 8),
                },
            )
            self.add_guidance(
                result=self.output,
                column=col,
                phase="ml",
                level="warn",
                title="Near-Zero Variance - Minimal Signal",
                body=(
                    f"{col} has near-zero variance (std: {std:.4g}). Features with "
                    f"essentially no spread provide no discriminative power to any "
                    f"model and can cause numerical instability in algorithms that "
                    f"scale by variance (PCA, SVM, regularised regression). "
                    f"Drop before modelling unless the column is the target variable."
                ),
                actions=[
                    {
                        "action": "drop",
                        "column": col,
                        "detail": "Zero variance - no signal for any model",
                    },
                ],
                metric={"mean": round(mean, 6), "std": round(std, 6)},
            )

        # --- Mean / median divergence ---
        # Only fire when we have enough spread to make the gap meaningful.
        # Skip if near_zero_var already fired (redundant for effectively constant cols)
        # and skip if std is zero to avoid division errors.
        if (
            not near_zero_var
            and std is not None
            and std > 0
            and mean is not None
            and median is not None
        ):
            gap_in_std = abs(mean - median) / std
            gap_info = 0.5  # noticeable asymmetry
            gap_warn = 1.0  # substantial pull - mean is no longer representative

            if gap_in_std >= gap_info:
                level: Literal["info", "warn"] = (
                    "warn" if gap_in_std >= gap_warn else "info"
                )
                direction: Literal["above", "below"] = (
                    "above" if mean > median else "below"
                )
                pull_desc: Literal[
                    "downward by low values", "upward by high values"
                ] = (
                    "upward by high values"
                    if mean > median
                    else "downward by low values"
                )

                eda_body: str = (
                    f"The mean of {col} ({mean:.4g}) sits {direction} the median "
                    f"({median:.4g}) by {gap_in_std:.2f} standard deviations. "
                    f"The mean is being pulled {pull_desc}. "
                    f"{
                        'For most analytical purposes the median is a more '
                        'representative centre for this column. '
                        if level == 'warn'
                        else ''
                    }"
                    f"Check the histogram to see whether the asymmetry comes from "
                    f"a long tail or from a cluster of outliers at one end."
                )
                ml_body: str = (
                    f"{col} has a mean–median gap of {gap_in_std:.2f} standard "
                    f"deviations, indicating an asymmetric distribution. "
                    f"Models that assume normality (linear regression, LDA, "
                    f"Gaussian NB) will be affected. See the skewness findings "
                    f"for specific transform recommendations."
                )

                self.add_guidance(
                    result=self.output,
                    column=col,
                    phase="eda",
                    level=level,
                    title=f"Mean-Median Gap ({gap_in_std:.2f}σ)",  # noqa: RUF001
                    body=eda_body.strip(),
                    actions=[],
                    metric={
                        "mean": round(mean, 4),
                        "median": round(median, 4),
                        "std": round(std, 4),
                        "gap_in_std": round(gap_in_std, 4),
                    },
                )
                self.add_guidance(
                    result=self.output,
                    column=col,
                    phase="ml",
                    level=level,
                    title=f"Distributional Asymmetry ({gap_in_std:.2f}σ gap)",
                    body=ml_body.strip(),
                    actions=[
                        {
                            "action": "see_task",
                            "task": "detect_skewness",
                            "column": col,
                            "detail": "Skewness task provides specific transform "
                            "recommendations",
                        },
                    ],
                    metric={
                        "mean": round(mean, 4),
                        "median": round(median, 4),
                        "std": round(std, 4),
                        "gap_in_std": round(gap_in_std, 4),
                    },
                )
