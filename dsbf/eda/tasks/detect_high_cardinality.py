# dsbf/eda/tasks/detect_high_cardinality.py

from dsbf.utils.backend import is_polars


def detect_high_cardinality(df, threshold=50):
    results = {}
    if is_polars(df):
        for col in df.columns:
            try:
                n_unique = df[col].n_unique()
                if n_unique > threshold:
                    results[col] = n_unique
            except Exception:
                continue
    else:
        for col in df.columns:
            try:
                n_unique = df[col].nunique()
                if n_unique > threshold:
                    results[col] = n_unique
            except Exception:
                continue
    return results
