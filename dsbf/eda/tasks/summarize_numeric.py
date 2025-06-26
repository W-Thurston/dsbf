# dsbf/eda/tasks/summarize_numeric.py

from dsbf.utils.backend import is_polars


def summarize_numeric(df):
    if is_polars(df):
        return df.describe().to_pandas().to_dict()
    return df.describe().to_dict()
