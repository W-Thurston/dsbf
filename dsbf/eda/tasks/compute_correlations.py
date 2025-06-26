# dsbf/eda/tasks/compute_correlations.py

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from dsbf.core.base_task import BaseTask
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
            if is_polars(df):
                df = df.to_pandas()

            correlations: Dict[str, float] = {}

            # Numeric correlation matrix using Pearson
            numeric_cols = df.select_dtypes(include=np.number).columns
            for i, col1 in enumerate(numeric_cols):
                for j in range(i + 1, len(numeric_cols)):
                    col2 = numeric_cols[j]
                    corr = df[col1].corr(df[col2])
                    correlations[f"{col1}|{col2}"] = corr

            # Categorical correlation matrix using Cramér’s V
            cat_cols = df.select_dtypes(include="object").columns
            for i, col1 in enumerate(cat_cols):
                for j in range(i + 1, len(cat_cols)):
                    col2 = cat_cols[j]
                    v = cramers_v(df[col1], df[col2])
                    correlations[f"{col1}|{col2}"] = v

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Computed correlations for {len(correlations)} column pairs.",
                data=correlations,
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during correlation computation: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
