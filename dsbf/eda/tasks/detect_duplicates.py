# dsbf/eda/tasks/detect_duplicates.py

from dsbf.utils.backend import is_polars


def detect_duplicates(df):
    if is_polars(df):
        return df.shape[0] - df.unique().shape[0]
    return df.duplicated().sum()
