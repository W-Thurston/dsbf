# dsbf/utils/backend.py


def is_polars(df):
    return df.__class__.__module__.startswith("polars")


def is_text_polars(column):
    import polars as pl

    return column.dtype in [pl.Utf8, pl.Categorical]


def is_text_pandas(series):
    import pandas as pd

    non_null = series.dropna()
    return (
        pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or (series.dtype == object and non_null.map(type).eq(str).mean() > 0.8)
    )
