# dsbf/eda/tasks/summarize_unique.py

from dsbf.utils.backend import is_polars


def summarize_unique(df):
    if is_polars(df):
        return {col: df[col].n_unique() for col in df.columns}
    return df.nunique().to_dict()
