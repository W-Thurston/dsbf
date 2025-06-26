# dsbf/eda/tasks/detect_id_columns.py

from dsbf.utils.backend import is_polars


def detect_id_columns(df):
    results = {}
    try:
        n_rows = df.shape[0]
        if is_polars(df):
            for col in df.columns:
                try:
                    n_unique = df[col].n_unique()
                    if n_unique >= 0.95 * n_rows:
                        results[col] = f"{n_unique} unique values (likely ID)"
                except Exception:
                    continue
        else:
            for col in df.columns:
                try:
                    n_unique = df[col].nunique()
                    if n_unique >= 0.95 * n_rows:
                        results[col] = f"{n_unique} unique values (likely ID)"
                except Exception:
                    continue
    except Exception:
        results["error"] = "Could not evaluate ID columns"
    return results
