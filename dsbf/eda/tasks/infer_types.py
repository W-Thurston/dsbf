# dsbf/eda/tasks/infer_types.py

from typing import Any, Dict

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars


class InferTypes(BaseTask):
    """
    Infers data types for each column and assigns a smart tag
    (e.g., binary, numeric, datetime, boolean, string, etc.).
    """

    def run(self) -> None:
        try:
            df: Any = self.input_data

            # Ensure we're operating on a Pandas DataFrame
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
                    tag = "binary" if len(unique_vals) == 2 else "numeric"

                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    tag = "datetime"

                elif pd.api.types.is_string_dtype(df[col]):
                    # Try parsing to datetime to detect patterns like ISO strings
                    try:
                        parsed = pd.to_datetime(df[col], errors="coerce")
                        if parsed.notnull().mean() > 0.9:
                            tag = "likely_datetime_string"
                        else:
                            tag = "string"
                    except Exception:
                        tag = "string"

                results[col] = {"dtype": dtype, "tag": tag}

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Inferred types and tags for {len(results)} columns.",
                data=results,
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during type inference: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
