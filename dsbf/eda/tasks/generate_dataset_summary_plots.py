# dsbf/eda/tasks/generate_dataset_summary_plots.py

from typing import Any

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory

# Colour map for the dtype stacked-bar chart.
#
# Keys are pandas dtype.name strings (the inferred_dtype values that become
# bar segments).  Colours are drawn from the dashboard palette so the chart
# feels native rather than using Plotly defaults.
#
# Grouping logic:
#   Integer types  → blue family   (numeric, exact)
#   Float types    → indigo/violet (numeric, approximate)
#   String/object  → purple        (text-like)
#   Category       → fuchsia       (structured text)
#   Boolean        → emerald       (binary)
#   Datetime types → amber         (temporal)
#   Fallback/mixed → slate grey    (unknown / mixed)
DTYPE_COLOR_MAP: dict[str, str] = {
    # ── Integer variants ───────────────────────────────────────────────────
    "int8": "#38bdf8",  # sky-400
    "int16": "#38bdf8",
    "int32": "#60a5fa",  # blue-400  (primary accent)
    "int64": "#60a5fa",
    "Int8": "#38bdf8",  # nullable integer (pandas ExtensionType)
    "Int16": "#38bdf8",
    "Int32": "#60a5fa",
    "Int64": "#60a5fa",
    "uint8": "#7dd3fc",  # sky-300
    "uint16": "#7dd3fc",
    "uint32": "#93c5fd",  # blue-300
    "uint64": "#93c5fd",
    # ── Float variants ─────────────────────────────────────────────────────
    "float16": "#a78bfa",  # violet-400
    "float32": "#a78bfa",
    "float64": "#818cf8",  # indigo-400
    "Float32": "#a78bfa",  # nullable float
    "Float64": "#818cf8",
    # ── String / object ────────────────────────────────────────────────────
    "object": "#c084fc",  # purple-400
    "str": "#c084fc",
    "string": "#c084fc",  # pd.StringDtype
    # ── Category ───────────────────────────────────────────────────────────
    "category": "#e879f9",  # fuchsia-400
    # ── Boolean ────────────────────────────────────────────────────────────
    "bool": "#34d399",  # emerald-400
    "boolean": "#34d399",  # pd.BooleanDtype
    # ── Datetime / timedelta ───────────────────────────────────────────────
    "datetime64": "#fbbf24",  # amber-400
    "datetime64[ns]": "#fbbf24",
    "datetime64[us]": "#fbbf24",
    "datetime64[ms]": "#fbbf24",
    "datetime64[s]": "#fbbf24",
    "datetime64[ns, UTC]": "#f59e0b",  # amber-500
    "timedelta64": "#fb923c",  # orange-400
    "timedelta64[ns]": "#fb923c",
    # ── Complex ────────────────────────────────────────────────────────────
    "complex64": "#94a3b8",  # slate-400
    "complex128": "#94a3b8",
    # ── Unknown / mixed / fallback ─────────────────────────────────────────
    "unknown": "#475569",  # slate-600
    "mixed": "#475569",
    "object_": "#c084fc",  # alias occasionally seen
}


@register_task(
    display_name="Generate Dataset-Level Plots",
    description="Generates summary plots describing the dataset as a whole.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="any",
    domain="core",
    runtime_estimate="fast",
    tags=["visualization", "plotting", "dataset"],
)
class GenerateDatasetSummaryPlots(BaseTask):
    def run(self) -> None:
        try:
            df: Any = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            results: dict[str, dict[str, str]] = {}

            # --- Correlation Matrix (static + interactive) ---
            corr_static = PlotFactory.plot_correlation_static(
                df, save_path=self.get_output_path("correlation_matrix.png")
            )
            corr_interactive = PlotFactory.plot_correlation_interactive(
                df, json_path=self.get_output_path("correlation_matrix.json")
            )
            results["correlation_matrix"] = {
                "static": corr_static["path"],
                "interactive": corr_interactive.get("interactive", {}),
            }
            self._log("    Correlation Matrix Plotted.", "debug")

            # --- Null Matrix (binary heatmap) ---
            static_path = self.get_output_path("null_matrix.png")
            _ = PlotFactory.plot_null_matrix_static(df, static_path)  # null_result
            results["null_matrix"] = {"static": str(static_path)}

            interactive_path = self.get_output_path("null_matrix.json")
            PlotFactory.plot_null_matrix_interactive(df, json_path=interactive_path)
            results["null_matrix"]["interactive"] = str(interactive_path)
            self._log("    Null Matrix Plotted.", "debug")

            # --- Missingness Matrix via missingno ---
            static_path = self.get_output_path("missingness_matrix.png")
            _ = PlotFactory.plot_missingness_matrix(df, static_path)  # missingno_result
            results["missingness_matrix"] = {"static": str(static_path)}

            # interactive_path = self.get_output_path("missingness_matrix.json")
            # PlotFactory.plot_missing_matrix_interactive(df,json_path=interactive_path)
            # results["missingness_matrix"]["interactive"] = str(interactive_path)
            self._log("    Missingness Matrix Plotted.", "debug")

            # --- Intent vs. Inferred Dtype Mapping ---
            if self.context:
                col_type_info = self.context.get_metadata("semantic_types", {}) or {}
                inferred_types = self.context.get_metadata("inferred_dtypes", {}) or {}
                valid_cols = [col for col in col_type_info if col in df.columns]
                if valid_cols:
                    mapping_dict = {
                        col: {
                            "inferred_dtype": inferred_types.get(col, "unknown"),
                            "analysis_intent_dtype": col_type_info[col],
                        }
                        for col in valid_cols
                    }

                    mapping_df = pd.DataFrame.from_dict(mapping_dict, orient="index")
                    mapping_df = mapping_df[["inferred_dtype", "analysis_intent_dtype"]]

                    dtype_bar = PlotFactory.plot_stacked_bar_interactive(
                        mapping_df,
                        json_path=self.get_output_path("dtype_stacked_bar.json"),
                        title="Dtype Mapping: Inferred within Intent",
                        color_map=DTYPE_COLOR_MAP,
                    )
                    results["dtype_stacked_bar"] = {
                        "interactive": dtype_bar.get("interactive", {}),
                    }

                    self._log("    Infer Types Stacked Bar Plotted.", "debug")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Generated {len(results)} dataset-level plots."},
                data=results,
                plots={},  # plots handled inline
            )

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"[{self.name}] Task failed: {type(e).__name__} - {e}", level="warn"
            )
            self.output = make_failure_result(self.name, e)
