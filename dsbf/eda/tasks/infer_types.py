# dsbf/eda/tasks/infer_types.py

from dsbf.utils.backend import is_polars


def infer_types(df):
    if is_polars(df):
        return {col: str(df[col].dtype) for col in df.columns}
    return {col: str(dtype) for col, dtype in df.dtypes.items()}
