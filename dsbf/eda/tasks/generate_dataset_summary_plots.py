# dsbf/eda/tasks/generate_dataset_summary_plots.py

from typing import TYPE_CHECKING, Any

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory

if TYPE_CHECKING:
    from polars import DataFrame

DTYPE_COLOR_MAP: dict[str, str] = {
    "int64": "#1f77b4",
    "float64": "#17becf",
    "string": "#9467bd",
    "object": "#9467bd",
    "bool": "#2ca02c",
    "datetime": "#ff7f0e",
    "category": "#8c564b",
    "unknown": "#475569",
    "mixed": "#475569",
    "object_": "#c084fc",
}

DatasetPlotResults = dict[str, dict[str, Any]]


@register_task(
    display_name="Generate Dataset-Level Plots",
    description="Generates summary plots describing the dataset as a whole.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="any",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["visualization", "plotting", "dataset"],
)
class GenerateDatasetSummaryPlots(BaseTask):
    """
    Generate centralized dataset-level summary visualizations.

    This task owns all plots that describe the dataset as a whole rather than
    individual columns. It is the authoritative source for:

    - Correlation matrix (numeric column pairs, Pearson)
    - Null matrix (missingness pattern across rows and columns)
    - Missingness matrix (missingno-style heatmap)
    - Dtype stacked bar (inferred vs intent type breakdown per column)

    Individual column plots are owned by ``generate_univariate_plots``.
    Bivariate scatter and distribution plots are rendered on demand by the
    Relationships tab frontend.

    Results are stored in ``TaskResult.data`` keyed by plot name. Each entry
    contains ``static`` (file path string) and/or ``interactive`` (JSON structure)
    artifact references consumed by the Vue dashboard via the figures API.
    """

    def run(self) -> None:
        """
        Generate dataset-level summary plots and store artifact references.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df: pd.DataFrame = self._to_pandas(self.input_data)
            results: DatasetPlotResults = {}

            results["correlation_matrix"] = self._build_correlation_plot(df)
            results["null_matrix"] = self._build_null_matrix_plot(df)
            results["missingness_matrix"] = self._build_missingness_matrix_plot(df)

            dtype_mapping_plot: dict[str, Any] = self._build_dtype_mapping_plot(df)
            if dtype_mapping_plot:
                results["dtype_stacked_bar"] = dtype_mapping_plot

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Generated {len(results)} dataset-level plots."},
                data=results,
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

    def _build_correlation_plot(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Build static and interactive correlation matrix artifacts.

        Only numeric columns are included. Returns an empty dict when fewer
        than 2 numeric columns exist or the correlation matrix is empty.

        Args:
            df: Source pandas DataFrame.

        Returns:
            Dict with ``static`` and ``interactive`` artifact references,
            or empty dict if the plot cannot be generated.

        """
        numeric_df: DataFrame = df.select_dtypes(include=["number"])

        if numeric_df.shape[1] < 2:  # noqa: PLR2004
            return {}

        corr: DataFrame = numeric_df.corr()
        if corr.empty:
            return {}

        # Pass numeric_df (not df) — the correlation plot covers numeric columns only.
        static_result: dict[str, Any] = PlotFactory.plot_correlation_static(
            numeric_df,
            save_path=self.get_output_path("correlation_matrix.png"),
        )
        interactive_result: dict[str, Any] = PlotFactory.plot_correlation_interactive(
            numeric_df,
            json_path=self.get_output_path("correlation_matrix.json"),
        )
        self._log("Correlation matrix plotted.", "debug")
        return {
            "static": static_result.get("path", {}),
            "interactive": interactive_result.get("interactive", {}),
        }

    def _build_null_matrix_plot(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Build static and interactive null-matrix artifacts.

        The null matrix shows which cells are null vs non-null across the
        full dataset, useful for spotting row-level or column-level patterns.

        Args:
            df: Source pandas DataFrame.

        Returns:
            Dict with ``static`` and ``interactive`` artifact path strings.

        """
        static_path: str = self.get_output_path("null_matrix.png")
        PlotFactory.plot_null_matrix_static(df, static_path)

        interactive_path: str = self.get_output_path("null_matrix.json")
        PlotFactory.plot_null_matrix_interactive(df, json_path=interactive_path)

        self._log("Null matrix plotted.", "debug")
        return {
            "static": str(static_path),
            "interactive": str(interactive_path),
        }

    def _build_missingness_matrix_plot(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Build the static missingno-style missingness matrix artifact.

        Distinct from the null matrix — uses the missingno library to render
        a sorted, dendrogram-clustered view of missingness patterns.

        Args:
            df: Source pandas DataFrame.

        Returns:
            Dict with ``static`` artifact path string.

        """
        static_path: str = self.get_output_path("missingness_matrix.png")
        PlotFactory.plot_missingness_matrix(df, static_path)
        self._log("Missingness matrix plotted.", "debug")
        return {"static": str(static_path)}

    def _build_dtype_mapping_plot(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Build the stacked dtype mapping artifact when context metadata is available.

        Visualises the relationship between inferred storage dtypes and DSBF
        analysis-intent types (e.g. how many int64 columns are treated as
        continuous vs categorical).

        Args:
            df: Source pandas DataFrame.

        Returns:
            Dict with ``interactive`` artifact reference, or empty dict if
            semantic type metadata is unavailable.

        """
        if not self.context:
            return {}

        semantic_types: dict = self.context.get_metadata("semantic_types", {}) or {}
        inferred_dtypes: dict = self.context.get_metadata("inferred_dtypes", {}) or {}
        valid_columns: list[str] = [col for col in semantic_types if col in df.columns]

        if not valid_columns:
            return {}

        mapping_df: DataFrame = pd.DataFrame.from_dict(
            {
                col: {
                    "inferred_dtype": inferred_dtypes.get(col, "unknown"),
                    "analysis_intent_dtype": semantic_types[col],
                }
                for col in valid_columns
            },
            orient="index",
        )[["inferred_dtype", "analysis_intent_dtype"]]

        interactive_result: dict[str, Any] = PlotFactory.plot_stacked_bar_interactive(
            mapping_df,
            json_path=self.get_output_path("dtype_stacked_bar.json"),
            title="Dtype Mapping: Inferred within Intent",
            color_map=DTYPE_COLOR_MAP,
        )
        self._log("Dtype stacked bar plotted.", "debug")
        return {"interactive": interactive_result.get("interactive", {})}
