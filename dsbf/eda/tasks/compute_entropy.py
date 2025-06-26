# dsbf/eda/tasks/compute_entropy.py

from dsbf.utils.backend import is_polars


def compute_entropy(df):
    if is_polars(df):
        from math import log2

        results = {}
        for col in df.columns:
            if df[col].dtype == "str":
                counts = df[col].value_counts()
                total = counts["counts"].sum()
                probs = [count / total for count in counts["counts"]]
                entropy = -sum(p * log2(p) for p in probs if p > 0)
                results[col] = entropy
        return results
    else:
        from scipy.stats import entropy

        results = {}
        for col in df.select_dtypes(include="object").columns:
            counts = df[col].value_counts()
            results[col] = entropy(counts, base=2)
        return results
