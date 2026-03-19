# dsbf/eda/tasks/generate_univariate_plots.py

from typing import TYPE_CHECKING, Any

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory

if TYPE_CHECKING:
    from polars import Series

ColumnPlotResults = dict[str, dict[str, Any]]


@register_task(
    display_name="Generate Univariate Plots",
    description="Generates univariate plots based on inferred semantic types.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="any",
    phase="eda",
    domain="core",
    runtime_estimate="moderate",
    tags=["visualization", "plotting", "univariate"],
    expected_semantic_types=["any"],
)
class GenerateUnivariatePlots(BaseTask):
    """
    Generate centralized per-column univariate plots.

    This task is the authoritative source for all column-level visualizations.
    It consumes the ``semantic_types`` metadata written by ``infer_types`` and
    generates the appropriate plot type per column:

    - **categorical** columns: bar plot (frequency by value)
    - **continuous** columns: histogram, boxplot, and composite (boxplot + histogram)

    Results are stored in ``TaskResult.data`` keyed by column name, then by plot
    type (``bar``, ``histogram``, ``boxplot``, ``composite``). Each entry contains
    ``static`` (file path) and/or ``interactive`` (JSON structure) artifact
    references consumed by the Vue dashboard's Distributions tab.

    Dataset-level plots (correlation matrix, missingness matrix, dtype breakdown)
    are owned by ``generate_dataset_summary_plots``.
    """

    def run(self) -> None:
        """
        Generate univariate plots for categorical and continuous columns.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df: pd.DataFrame = self._to_pandas(self.input_data)
            semantic_types: dict[str, str] = self._get_semantic_types()
            results: dict[str, ColumnPlotResults] = {}

            for column, intent in semantic_types.items():
                if column not in df.columns or intent not in {
                    "categorical",
                    "continuous",
                }:
                    continue

                series: Series = df[column].dropna()
                if series.empty:
                    continue

                if intent == "categorical":
                    results[column] = self._build_categorical_plots(series)
                else:
                    results[column] = self._build_continuous_plots(series, column)

                self._log(
                    f"    '{column}' plotted: {list(results[column].keys())}",
                    "debug",
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Generated univariate plots for {len(results)} columns."
                    ),
                },
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

    def _get_semantic_types(self) -> dict[str, str]:
        """
        Return semantic type metadata from the analysis context.

        Returns:
            Dict mapping column name → DSBF semantic type string, or empty dict
            if no context or no metadata is available.

        """
        if not self.context:
            self._log(
                (
                    "    [GenerateUnivariatePlots] No context — semantic types "
                    "unavailable."
                ),
                "debug",
            )
            return {}
        metadata: dict = self.context.get_metadata("semantic_types", {}) or {}
        return {str(col): str(intent) for col, intent in metadata.items()}

    def _build_categorical_plots(self, series: pd.Series) -> ColumnPlotResults:
        """
        Build bar plots for a categorical series.

        Args:
            series: Non-null values for the column (pandas Series).

        Returns:
            Dict with ``bar`` key containing ``static`` and ``interactive``
            artifact references.

        """
        static_result: dict[str, Any] = PlotFactory.plot_barplot_static(
            series,
            self.get_output_path(f"{series.name}_bar.png"),
            top_k=100,
        )
        interactive_result: dict[str, Any] = PlotFactory.plot_barplot_interactive(
            series,
            json_path=self.get_output_path(f"{series.name}_bar.json"),
            title=f"{series.name} - Frequency Plot",
            top_k=100,
        )
        return {
            "bar": {
                "static": static_result.get("path", {}),
                "interactive": interactive_result.get("interactive", {}),
            },
        }

    def _build_continuous_plots(
        self,
        series: pd.Series,
        column: str,
    ) -> ColumnPlotResults:
        """
        Build histogram, boxplot, and composite views for a continuous series.

        Args:
            series: Non-null values for the column (pandas Series).
            column: Column name used for output file naming.

        Returns:
            Dict with ``histogram``, ``boxplot``, and optionally ``composite``
            keys, each containing ``static`` and/or ``interactive`` artifact
            references.

        """
        results: ColumnPlotResults = {}

        histogram_static: dict[str, Any] = PlotFactory.plot_histogram_static(
            series,
            self.get_output_path(f"{column}_hist.png"),
        )
        histogram_interactive: dict[str, Any] = PlotFactory.plot_histogram_interactive(
            series,
            json_path=self.get_output_path(f"{column}_hist.json"),
            title=f"{column} - Histogram",
        )
        results["histogram"] = {
            "static": histogram_static.get("path", {}),
            "interactive": histogram_interactive.get("interactive", {}),
        }

        boxplot_static: dict[str, Any] = PlotFactory.plot_boxplot_static(
            series,
            self.get_output_path(f"{column}_boxplot.png"),
        )
        boxplot_interactive: dict[str, Any] = PlotFactory.plot_boxplot_interactive(
            series,
            json_path=self.get_output_path(f"{column}_boxplot.json"),
            title=f"{column} - Boxplot",
        )
        results["boxplot"] = {
            "static": boxplot_static.get("path", {}),
            "interactive": boxplot_interactive.get("interactive", {}),
        }

        composite_static: dict[str, Any] = (
            PlotFactory.plot_boxplot_hist_composite_static(
                series,
                save_path=self.get_output_path(f"{column}_composite.png"),
            )
        )
        if composite_static:
            results["composite"] = {"static": composite_static}
        else:
            self._log(
                f"    '{column}' composite plot skipped (PlotFactory returned empty).",
                "debug",
            )

        return results
