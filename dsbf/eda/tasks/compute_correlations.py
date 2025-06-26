# dsbf/eda/tasks/compute_correlations.py

import numpy as np

from dsbf.utils.backend import is_polars


def compute_correlations(df):
    if is_polars(df):
        numeric_cols = [
            col for col, dtype in zip(df.columns, df.dtypes) if dtype in ("i64", "f64")
        ]
        df_num = df.select(numeric_cols)
        corr = {}
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if j >= i:
                    value = df_num.select([col1, col2]).pearson_corr(col1, col2)
                    corr[(col1, col2)] = value
        return corr
    else:
        numeric_df = df.select_dtypes(include=np.number)
        corr_matrix = numeric_df.corr(method="pearson")
        return {
            (col1, col2): corr_matrix.loc[col1, col2]
            for col1 in corr_matrix.columns
            for col2 in corr_matrix.columns
            if corr_matrix.loc[col1, col2] != 1.0
        }
