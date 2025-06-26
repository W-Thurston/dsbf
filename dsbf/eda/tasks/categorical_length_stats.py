# dsbf/core/eda/tasks/categorical_length_stats.py

import polars as pl

from dsbf.utils.backend import is_polars, is_text_pandas, is_text_polars


def categorical_length_stats(df):
    results = {}
    if is_polars(df):
        for col in df.columns:
            if is_text_polars(df[col]):
                try:
                    lengths = df.select(
                        pl.col(col).cast(pl.Utf8).str.len_chars().alias("len")
                    ).drop_nulls()["len"]
                    if lengths.len() > 0:
                        results[col] = {
                            "mean_length": lengths.mean(),
                            "max_length": lengths.max(),
                            "min_length": lengths.min(),
                        }
                except Exception as e:
                    print(
                        f"[categorical_length_stats]"
                        f"Error processing Polars column {col}: {e}"
                    )
                    continue
    else:
        for col in df.columns:
            if is_text_pandas(df[col]):
                try:
                    lengths = df[col].dropna().astype(str).map(len)
                    if len(lengths) > 0:
                        results[col] = {
                            "mean_length": lengths.mean(),
                            "max_length": lengths.max(),
                            "min_length": lengths.min(),
                        }
                except Exception as e:
                    print(
                        f"[categorical_length_stats]"
                        f"Error processing Pandas column {col}: {e}"
                    )
                    continue
    return results
