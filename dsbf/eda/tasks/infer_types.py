# dsbf/eda/tasks/infer_types.py

from typing import Any, Dict

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Infer Column Types",
    description="Infers column types (e.g., categorical, numeric, text) heuristically.",
    depends_on=[],
    profiling_depth="basic",
    stage="raw",
    tags=["typing", "metadata"],
)
class InferTypes(BaseTask):
    """
    Infers data types for each column and assigns a smart tag
    (e.g., binary, numeric, datetime, boolean, string, etc.).
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

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
                        parsed = pd.to_datetime(
                            df[col], errors="coerce", format="ISO8601"
                        )
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
                summary={
                    "message": (f"Inferred types and tags for {len(results)} columns.")
                },
                data=results,
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
