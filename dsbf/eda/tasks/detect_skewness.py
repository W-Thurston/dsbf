# dsbf/eda/tasks/detect_skewness.py

from dsbf.utils.backend import is_polars


def detect_skewness(df):
    if is_polars(df):
        import numpy as np

        skewness = {}
        for col in df.columns:
            if df[col].dtype in ("i64", "f64"):
                series = df[col].to_numpy()
                mean = np.mean(series)
                std = np.std(series)
                if std != 0:
                    skew = np.mean(((series - mean) / std) ** 3)
                    skewness[col] = skew
        return skewness
    else:
        from scipy.stats import skew

        numeric = df.select_dtypes(include="number")
        return {col: skew(numeric[col].dropna()) for col in numeric.columns}
