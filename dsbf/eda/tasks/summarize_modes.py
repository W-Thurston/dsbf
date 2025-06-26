# dsbf/eda/tasks/summarize_modes.py

from dsbf.utils.backend import is_polars


def summarize_modes(df):
    if is_polars(df):
        return {col: df[col].mode().to_list() for col in df.columns}
    return df.mode().iloc[0].to_dict()
