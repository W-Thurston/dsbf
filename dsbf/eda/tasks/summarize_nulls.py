# dsbf/eda/tasks/summarize_nulls.py

from dsbf.utils.backend import is_polars


def summarize_nulls(df):
    if is_polars(df):
        return {col: str(df[col].null_count()) for col in df.columns}
    return df.isnull().sum().to_dict()
