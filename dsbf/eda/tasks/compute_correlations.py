# dsbf/eda/tasks/compute_correlations.py

from typing import Dict

import numpy as np
import pandas as pd
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
from dsbf.utils.plot_factory import PlotFactory


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    return np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else 0.0


@register_task(
    display_name="Compute Correlations",
    description="Calculates Pearson/Spearman correlations between numeric columns.",
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="moderate",
    tags=["numeric", "correlation"],
    expected_semantic_types=["continuous", "categorical"],
)
class ComputeCorrelations(BaseTask):
    def run(self) -> None:
        try:
            df = self.input_data
            correlations: Dict[str, float] = {}
            backend_used = "polars" if is_polars(df) else "pandas"

            # --- Polars numeric correlation ---
            if is_polars(df):
                import polars as pl

                try:
                    numeric_cols = [
                        col
                        for col in df.columns
                        if df[col].dtype in (pl.Float64, pl.Int64)
                    ]
                    numeric_df = df.select(numeric_cols)
                    if len(numeric_cols) >= 2:
                        corr_df = numeric_df.corr()
                        for i, col1 in enumerate(numeric_cols):
                            for j in range(i + 1, len(numeric_cols)):
                                col2 = numeric_cols[j]
                                value = corr_df.select(f"{col1}_{col2}").item()
                                correlations[f"{col1}|{col2}"] = value
                    else:
                        self._log(
                            "    Not enough numeric columns for correlation matrix.",
                            "debug",
                        )
                except Exception as e:
                    self._log(
                        f"    Polars correlation failed: {e}. Falling back to Pandas.",
                        "debug",
                    )
                    df = df.to_pandas()
                    backend_used = "pandas"

            # --- Pandas numeric correlation ---
            if not is_polars(df):
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) < 2:
                    self._log(
                        "    Fewer than 2 numeric columns —"
                        " skipping correlation computation."
                    )
                    self.output = TaskResult(
                        name=self.name,
                        status="success",
                        summary={
                            "message": (
                                "Not enough numeric columns to compute correlations."
                            )
                        },
                        data={},
                        plots={},
                    )
                    return
                numeric_df = df[numeric_cols]
                for i, col1 in enumerate(numeric_cols):
                    for j in range(i + 1, len(numeric_cols)):
                        col2 = numeric_cols[j]
                        corr = df[col1].corr(df[col2])
                        correlations[f"{col1}|{col2}"] = corr

            # --- Categorical Cramér’s V correlations ---
            if is_polars(df):
                df = df.to_pandas()
                backend_used = "mixed"

            cat_cols = df.select_dtypes(include="object").columns
            for i, col1 in enumerate(cat_cols):
                for j in range(i + 1, len(cat_cols)):
                    col2 = cat_cols[j]
                    v = cramers_v(df[col1], df[col2])
                    correlations[f"{col1}|{col2}"] = v

            result = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed correlations for "
                        f"{len(correlations)} column pairs."
                    )
                },
                data=correlations,
                metadata={
                    "backend": backend_used,
                    "numeric_pair_count": sum(
                        "|" in k and k.split("|")[0] in df.columns for k in correlations
                    ),
                    "categorical_pair_count": sum(
                        "|" in k and k.split("|")[0] in cat_cols for k in correlations
                    ),
                },
            )

            # --- Add reliability warnings from precomputed flags ---
            flags = self.ensure_reliability_flags()

            if flags["low_row_count"]:
                add_reliability_warning(
                    result,
                    level="strong_warning",
                    code="low_row_count",
                    description=(
                        "Pearson correlation may be statistically"
                        " unreliable with fewer than 30 observations."
                    ),
                    recommendation=(
                        "Use bootstrapped confidence intervals" " or collect more data."
                    ),
                )

            if flags["zero_variance_cols"]:
                add_reliability_warning(
                    result,
                    level="strong_warning",
                    code="zero_variance",
                    description=(
                        "The following features have near-zero variance: "
                        f"{flags['zero_variance_cols']}. "
                        "Correlation is undefined."
                    ),
                    recommendation=(
                        "Drop or impute constant features"
                        " before computing correlation."
                    ),
                )

            if flags["extreme_outliers"]:
                code = (
                    "extreme_outliers_low_n"
                    if flags["low_row_count"]
                    else "extreme_outliers"
                )
                description = (
                    "Some features contain extreme z-scores (|z| > 3),"
                    " but sample size is small (N < 30). "
                    "Outlier estimates may be unreliable."
                    if flags["low_row_count"]
                    else (
                        "Some features contain extreme z-scores (|z| > 3),"
                        " which may distort Pearson correlation."
                    )
                )
                recommendation = (
                    (
                        "Interpret outlier influence with caution"
                        " or validate using robust statistics."
                    )
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
                    (
                        "High skew was detected, but sample size is small (N < 30)."
                        " Skew estimates may be unstable."
                    )
                    if flags["low_row_count"]
                    else (
                        "One or more features are highly skewed,"
                        " which may distort correlation strength."
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

            # --- Optional visualization ---
            if correlations:
                corr_df = pd.DataFrame(
                    index=df.columns, columns=df.columns, dtype=float
                )
                for pair, value in correlations.items():
                    col1, col2 = pair.split("|")
                    corr_df.loc[col1, col2] = value
                    corr_df.loc[col2, col1] = value  # Ensure symmetry
                    corr_df.loc[col1, col1] = 1.0
                    corr_df.loc[col2, col2] = 1.0
                corr_df.fillna(1.0, inplace=True)

                # Convert to numeric-only (some non-numeric pairs may sneak in)
                numeric_corr = corr_df.select_dtypes(include=[np.number])

                save_path = self.get_output_path("correlation_heatmap.png")
                static_plot = PlotFactory.plot_correlation_static(
                    numeric_corr,
                    save_path=save_path,
                    title="Correlation Heatmap",
                )
                interactive_plot = PlotFactory.plot_correlation_interactive(
                    numeric_corr,
                    title="Correlation Heatmap",
                )
                result.plots = {
                    "correlation_matrix": {
                        "static": static_plot["path"],
                        "interactive": interactive_plot,
                    }
                }

            result.metadata.update(
                {
                    "suggested_viz_type": "heatmap",
                    "recommended_section": "Correlations",
                    "display_priority": "medium",
                    "column_types": self.get_column_type_info(list(df.columns)),
                }
            )

            log_reliability_warnings(self, result)
            self.output = result

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} — {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
