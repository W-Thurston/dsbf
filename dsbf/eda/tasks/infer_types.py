# dsbf/eda/tasks/infer_types.py

from typing import Any, Dict

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Infer Column Types",
    description="Infers both raw and analysis-intent dtypes for each column.",
    depends_on=[],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    tags=["typing", "metadata"],
)
class InferTypes(BaseTask):
    """
    Infers both the raw data type (e.g., 'int64', 'object') and the analysis-intent
    semantic type (e.g., 'categorical', 'continuous', 'text', 'id', 'datetime')
    for each column in the dataset.

    The inferred types are used to guide downstream tasks such as visualizations,
    statistical tests, or recommendations, while the analysis intent helps align
    profiling behavior with how the data will be used, not just what it is.
    """

    def run(self) -> None:
        """
        Executes the type inference process on the input DataFrame.
        Produces a mapping of column names to both raw and semantic types,
        and stores these in the analysis context for use by downstream tasks.
        """
        try:
            df: Any = self.input_data

            # Convert to Pandas for compatibility
            if is_polars(df):
                df = df.to_pandas()

            results: Dict[str, Dict[str, str]] = {}

            # Loop through each column to infer dtypes
            for col in df.columns:
                inferred_dtype: str = str(df[col].dtype)
                analysis_intent_dtype: str = "unknown"

                # Always record something, even for empty columns
                try:
                    series = df[col].dropna()
                    nunique = series.nunique()
                    total = series.size
                    uniq_ratio = nunique / total if total else 0

                    # ---- Heuristic rules for semantic typing ----
                    if pd.api.types.is_bool_dtype(series):
                        analysis_intent_dtype = "categorical"
                    elif pd.api.types.is_numeric_dtype(series):
                        if nunique == 2:
                            analysis_intent_dtype = "categorical"
                        elif uniq_ratio < 0.05 and nunique <= 20:
                            analysis_intent_dtype = "categorical"
                        else:
                            analysis_intent_dtype = "continuous"
                    elif pd.api.types.is_datetime64_any_dtype(series):
                        analysis_intent_dtype = "datetime"
                    elif pd.api.types.is_string_dtype(series):
                        try:
                            _ = pd.to_datetime(series, errors="raise", utc=True)
                            analysis_intent_dtype = "datetime"
                        except Exception:
                            if series.str.fullmatch(r"[A-Fa-f0-9\-]{8,}").mean() > 0.8:
                                analysis_intent_dtype = "id"
                            elif uniq_ratio > 0.9:
                                analysis_intent_dtype = "id"
                            elif series.str.len().mean() > 30:
                                analysis_intent_dtype = "text"
                            else:
                                analysis_intent_dtype = "categorical"
                except Exception:
                    pass  # Still record defaults below

                # Ensure every column is included
                results[col] = {
                    "inferred_dtype": inferred_dtype,
                    "analysis_intent_dtype": analysis_intent_dtype,
                }

            # ---- Store semantic type metadata in context ----
            if self.context:
                # Map: column → 'continuous' | 'categorical' | etc.
                self.context.set_metadata(
                    "semantic_types",
                    {
                        col: info["analysis_intent_dtype"]
                        for col, info in results.items()
                    },
                )
                # Map: column → pandas/polars dtype as string
                self.context.set_metadata(
                    "inferred_dtypes",
                    {col: info["inferred_dtype"] for col, info in results.items()},
                )

            # ---- Build and return TaskResult ----
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Inferred types for {len(results)} columns."},
                data=results,
            )

        except Exception as e:
            # On failure, return a standardized failure result unless debugging
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
