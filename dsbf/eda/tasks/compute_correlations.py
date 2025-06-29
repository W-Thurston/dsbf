# dsbf/eda/tasks/compute_correlations.py

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Compute Cramér's V (normalized chi-squared) between two categorical columns.

    Args:
        x (pd.Series): First categorical column.
        y (pd.Series): Second categorical column.

    Returns:
        float: Cramér's V score (0–1)
    """
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
    tags=["numeric", "correlation"],
)
class ComputeCorrelations(BaseTask):
    """
    Computes pairwise correlations for numeric and categorical columns.
    - Uses Pearson for numeric-numeric pairs.
    - Uses Cramér's V for categorical-categorical pairs.
    """

    def run(self) -> None:
        """
        Run the correlation task on input_data.
        Produces a TaskResult with a flat dictionary of correlations.
        """

        try:
            df = self.input_data
            correlations: Dict[str, float] = {}
            backend_used = "polars" if is_polars(df) else "pandas"

            if is_polars(df):
                import polars as pl

                try:
                    numeric_cols = [
                        col
                        for col in df.columns
                        if df[col].dtype in (pl.Float64, pl.Int64)
                    ]
                    if len(numeric_cols) >= 2:
                        corr_df = df.select(numeric_cols).corr()
                        # Parse Polars correlation matrix
                        for i, col1 in enumerate(numeric_cols):
                            for j in range(i + 1, len(numeric_cols)):
                                col2 = numeric_cols[j]
                                value = corr_df.select(f"{col1}_{col2}").item()
                                correlations[f"{col1}|{col2}"] = value
                        self._log(
                            (
                                f"Computed Polars correlation matrix for "
                                f"{len(numeric_cols)} numeric columns."
                            ),
                            "debug",
                        )
                    else:
                        self._log(
                            (
                                "Not enough numeric columns for correlation matrix;"
                                " skipping numeric pairwise correlations."
                            ),
                            "debug",
                        )
                except Exception as e:
                    self._log(
                        f"Polars correlation failed: {e}. Falling back to Pandas.",
                        "debug",
                    )
                    df = df.to_pandas()
                    backend_used = "pandas"

            if not is_polars(df):  # pandas fallback or already pandas
                numeric_cols = df.select_dtypes(include=np.number).columns
                for i, col1 in enumerate(numeric_cols):
                    for j in range(i + 1, len(numeric_cols)):
                        col2 = numeric_cols[j]
                        corr = df[col1].corr(df[col2])
                        correlations[f"{col1}|{col2}"] = corr

            # Always compute categorical correlations using Pandas (Cramér's V)
            if is_polars(df):
                df = df.to_pandas()
                backend_used = "mixed"

            cat_cols = df.select_dtypes(include="object").columns
            for i, col1 in enumerate(cat_cols):
                for j in range(i + 1, len(cat_cols)):
                    col2 = cat_cols[j]
                    v = cramers_v(df[col1], df[col2])
                    correlations[f"{col1}|{col2}"] = v

            self._log(
                f"Computed correlations for {len(correlations)} column pairs.", "info"
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed correlations for"
                        f" {len(correlations)} column pairs."
                    )
                },
                data=correlations,
                metadata={
                    "backend": backend_used,
                    "numeric_pair_count": sum(
                        "|" in k and k.split("|")[0] in df.columns
                        for k in correlations.keys()
                    ),
                    "categorical_pair_count": sum(
                        "|" in k and k.split("|")[0] in cat_cols
                        for k in correlations.keys()
                    ),
                },
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary={"message": f"Error during correlation computation: {e}"},
                data=None,
                metadata={"exception": type(e).__name__},
            )
