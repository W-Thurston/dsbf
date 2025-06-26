# dsbf/eda/tasks/compute_correlations.py

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def cramers_v(x, y) -> float:
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    return np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else 0.0


def compute_correlations(df: Any) -> TaskResult:
    """
    Computes pairwise correlations for numeric and categorical columns.
    Uses Pearson for numeric, Cramér's V for categorical-categorical pairs.

    Args:
        df (DataFrame): Input Polars or Pandas DataFrame.

    Returns:
        TaskResult: Correlation matrix as flat dictionary.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        correlations: Dict[str, float] = {}
        numeric_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include="object").columns

        # Pearson for numeric pairs
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if j > i:
                    corr = df[col1].corr(df[col2])
                    correlations[f"{col1}|{col2}"] = corr

        # Cramér's V for categorical pairs
        for i, col1 in enumerate(cat_cols):
            for j, col2 in enumerate(cat_cols):
                if j > i:
                    v = cramers_v(df[col1], df[col2])
                    correlations[f"{col1}|{col2}"] = v

        return TaskResult(
            name="compute_correlations",
            status="success",
            summary=f"Computed correlations for {len(correlations)} column pairs.",
            data=correlations,
        )

    except Exception as e:
        return TaskResult(
            name="compute_correlations",
            status="failed",
            summary=f"Error during correlation computation: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
