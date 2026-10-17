# dsbf/utils/reliability_stats.py

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation, skew


def compute_reliability_flags(df: pd.DataFrame) -> dict:
    """
    Compute global reliability diagnostics for a numeric dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with numeric columns.

    Returns:
        dict: Reliability flags and stats including skew, std, outliers, etc.
    """
    numeric_df = df.select_dtypes(include=np.number).dropna()
    n_rows = len(numeric_df)

    stds = numeric_df.std().to_dict()
    means = numeric_df.mean().to_dict()

    with np.errstate(invalid="ignore"):
        skew_vals = dict(
            zip(
                numeric_df.columns,
                skew(numeric_df, nan_policy="omit", bias=False),
            )
        )

    # MAD-based outlier detection
    mad = {
        col: median_abs_deviation(numeric_df[col], nan_policy="omit")
        for col in numeric_df.columns
    }
    mad = {k: (v if v != 0 else 1e-8) for k, v in mad.items()}
    robust_z = pd.DataFrame(
        {
            col: (numeric_df[col] - numeric_df[col].median()) / mad[col]
            for col in numeric_df.columns
        }
    )
    has_outliers = (robust_z.abs() > 3).any().any()

    low_var_cols = [col for col, std in stds.items() if std is not None and std < 1e-8]

    return {
        "n_rows": n_rows,
        "low_row_count": n_rows < 30,
        "extreme_outliers": has_outliers,
        "high_skew": any(abs(s) > 2 for s in skew_vals.values() if s is not None),
        "zero_variance_cols": low_var_cols,
        "skew_vals": skew_vals,
        "stds": stds,
        "means": means,
    }
