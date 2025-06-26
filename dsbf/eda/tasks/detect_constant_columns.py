# dsbf/eda/tasks/detect_constant_columns.py

from dsbf.utils.backend import is_polars


def detect_constant_columns(df):
    if is_polars(df):
        return [col for col in df.columns if df[col].n_unique() == 1]
    return [col for col in df.columns if df[col].nunique() == 1]
