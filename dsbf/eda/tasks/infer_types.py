# dsbf/eda/tasks/infer_types.py

from typing import Any, Dict

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


def infer_types(df: Any) -> TaskResult:
    """
    Infers the data types of each column, with extended smart tagging.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.

    Returns:
        TaskResult: Dictionary with raw types and smart tags.
    """
    try:
        if is_polars(df):
            df = df.to_pandas()

        results: Dict[str, Dict[str, str]] = {}

        for col in df.columns:
            dtype = str(df[col].dtype)
            tag = "unknown"

            if pd.api.types.is_bool_dtype(df[col]):
                tag = "boolean"
            elif pd.api.types.is_numeric_dtype(df[col]):
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2:
                    tag = "binary"
                else:
                    tag = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                tag = "datetime"
            elif pd.api.types.is_string_dtype(df[col]):
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notnull().mean() > 0.9:
                        tag = "likely_datetime_string"
                    else:
                        tag = "string"
                except Exception:
                    tag = "string"

            results[col] = {"dtype": dtype, "tag": tag}

        return TaskResult(
            name="infer_types",
            status="success",
            summary=f"Inferred types and tags for {len(results)} columns.",
            data=results,
        )

    except Exception as e:
        return TaskResult(
            name="infer_types",
            status="failed",
            summary=f"Error during type inference: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
