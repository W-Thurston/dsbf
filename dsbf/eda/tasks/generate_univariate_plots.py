# dsbf/eda/tasks/generate_univariate_plots.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Generate Univariate Plots",
    description="Generates Univariate plots based on inferred semantic types.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="any",
    domain="core",
    runtime_estimate="moderate",
    tags=["visualization", "plotting", "univariate"],
    expected_semantic_types=["any"],
)
class GenerateUnivariatePlots(BaseTask):
    def run(self) -> None:
        try:
            df: Any = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            if self.context:
                semantic_types = self.context.get_metadata("semantic_types", {}) or {}

            else:
                semantic_types = {}

            results: dict[str, dict[str, Any]] = {}

            for col, intent in semantic_types.items():
                if intent not in ("categorical", "continuous"):
                    continue

                col_data = df[col].dropna()
                if col_data.empty:
                    continue

                results[col] = {}

                if intent == "categorical":
                    # === Bar ===
                    # Static (Seaborn/Matplotlib)
                    bar_static = PlotFactory.plot_barplot_static(
                        col_data,
                        self.get_output_path(f"{col}_bar.png"),
                        top_k=100,
                    )

                    # Interactive (Plotly)
                    bar_interactive = PlotFactory.plot_barplot_interactive(
                        col_data,
                        json_path=self.get_output_path(f"{col}_bar.json"),
                        title=f"{col} - Frequency Plot",
                        top_k=100,
                    )

                    results[col]["bar"] = {
                        "static": bar_static["path"],
                        "interactive": bar_interactive.get("interactive", {}),
                    }

                elif intent == "continuous":
                    # === Histogram ===
                    # Static (Seaborn/Matplotlib)
                    hist_static = PlotFactory.plot_histogram_static(
                        col_data, self.get_output_path(f"{col}_hist.png")
                    )

                    # Interactive (Plotly)
                    hist_interactive = PlotFactory.plot_histogram_interactive(
                        col_data,
                        json_path=self.get_output_path(f"{col}_hist.json"),
                        title=f"{col} - Histogram",
                    )

                    results[col]["histogram"] = {
                        "static": hist_static["path"],
                        "interactive": hist_interactive.get("interactive", {}),
                    }

                    # === Boxplot ===
                    # Static (Seaborn/Matplotlib)
                    box_static = PlotFactory.plot_boxplot_static(
                        col_data, self.get_output_path(f"{col}_boxplot.png")
                    )

                    # Interactive (Plotly)
                    box_interactive = PlotFactory.plot_boxplot_interactive(
                        col_data,
                        json_path=self.get_output_path(f"{col}_boxplot.json"),
                        title=f"{col} - Boxplot",
                    )

                    results[col]["boxplot"] = {
                        "static": box_static["path"],
                        "interactive": box_interactive.get("interactive", {}),
                    }

                    # === Composite Hist/Box ===
                    # Static (Seaborn/Matplotlib)
                    composite_paths = PlotFactory.plot_boxplot_hist_composite_static(
                        col_data,
                        save_path=self.get_output_path(f"{col}_composite.png"),
                    )

                    if composite_paths:
                        # Ensure top-level entry is a dict
                        if not isinstance(results.get(col), dict):
                            results[col] = {}

                        # Ensure composite key is present and a dict
                        if not isinstance(results[col].get("composite"), dict):
                            results[col]["composite"] = {}

                        # Now safe to assign
                        results[col]["composite"] = {"static": composite_paths}

                self._log(
                    f"    Column `{col}` Plotted {list(results[col].keys())}", "debug"
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Generated univariate plots for {len(results)} columns."
                    )
                },
                data=results,
                plots={},  # plots now lives entirely inside `data`
            )

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"[{self.name}] Task failed: {type(e).__name__} - {e}", level="warn"
            )
            self.output = make_failure_result(self.name, e)
