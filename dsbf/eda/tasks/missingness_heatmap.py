# dsbf/eda/tasks/missingness_heatmap.py

from typing import Any

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    display_name="Missingness Heatmap",
    description=(
        "Summarizes dataset missingness for downstream centralized visualization."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="report",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["missing", "summary"],
    expected_semantic_types=["any"],
)
class MissingnessHeatmap(BaseTask):
    """
    Summarize dataset missingness without generating standalone plot artifacts.

    Plot generation for missingness patterns is handled centrally by
    ``generate_dataset_summary_plots`` (null matrix and missingno heatmap).
    This task computes lightweight missingness summary statistics that the
    data_quality_scorer and frontend can consume directly without reading
    the full TaskResult data from disk.

    Output includes:
    - ``missing_cells``: Total count of null cells across the dataset
    - ``missing_columns``: List of column names with at least one null
    - ``column_count`` / ``row_count``: Dataset shape
    """

    def run(self) -> None:
        """
        Compute dataset-level missingness counts and store in output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df: pd.DataFrame = self._to_pandas(self.input_data)
            matched_columns, excluded_columns = self.get_columns_by_intent()

            self._log(f"    Processing {len(matched_columns)} column(s)", "debug")

            missing_cells = int(df.isna().sum().sum())
            missing_columns: list[str] = [
                col for col in df.columns if df[col].isna().any()
            ]

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Found {missing_cells} missing cells in dataset."},
                data={
                    "missing_cells": missing_cells,
                    "missing_columns": missing_columns,
                    "column_count": int(df.shape[1]),
                    "row_count": int(df.shape[0]),
                },
                metadata={
                    "suggested_viz_type": "heatmap",
                    "recommended_section": "Missingness",
                    "display_priority": "medium",
                    "excluded_columns": excluded_columns,
                    "column_types": self.get_column_type_info(
                        matched_columns + list(excluded_columns.keys()),
                    ),
                },
            )
        except Exception as error:
            if self.context:
                raise
            self._log(
                f"Task failed: {type(error).__name__} - {error}",
                level="warn",
            )
            self.output = make_failure_result(self.name, error)

    def _to_pandas(self, data: Any) -> pd.DataFrame:
        """
        Convert a supported dataframe backend to pandas.

        Args:
            data: Input DataFrame (pandas or Polars).

        Returns:
            pandas DataFrame.

        Raises:
            TypeError: If data is neither pandas nor Polars.

        """
        if is_polars(data):
            return data.to_pandas()
        if isinstance(data, pd.DataFrame):
            return data
        msg: str = f"Expected pandas or polars dataframe, got {type(data).__name__}."
        raise TypeError(msg)
