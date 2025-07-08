# dsbf/eda/tasks/detect_skewness.py

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import skew

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    display_name="Detect Skewness",
    description="Computes skewness of numeric columns.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["distribution", "skew"],
    expected_semantic_types=["continuous"],
)
class DetectSkewness(BaseTask):
    """
    Computes skewness for all numeric columns in a dataset
    that have been semantically tagged as 'continuous'.

    Skewness quantifies the asymmetry of a distribution,
    and can be useful for identifying transformations
    or modeling assumptions that may need attention.
    """

    def run(self) -> None:
        try:
            df: Any = self.input_data
            skewness: Dict[str, float] = {}
            plots: dict[str, dict[str, Any]] = {}

            # Use semantic typing to select relevant columns
            numeric_cols, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(numeric_cols)} 'continuous' column(s)", "debug")

            # Compute skewness for Polars DataFrame
            if is_polars(df):
                df = df.select(numeric_cols) if numeric_cols else df
                for col in df.columns:
                    series = df[col].drop_nulls().to_numpy()
                    if series.size == 0:
                        self._log(f"{col} skipped: empty after dropna()", "debug")
                        continue
                    mean = np.mean(series)
                    std = np.std(series)
                    skew_val = (
                        float(np.mean(((series - mean) / std) ** 3))
                        if std != 0
                        else 0.0
                    )
                    skewness[col] = skew_val

                    # Ensure proper Series object for plotting
                    series = pd.Series(series, name=col)

                    annotations = [f"Skewness: {skew_val:.3f}"]
                    save_path = self.get_output_path(f"{col}_histogram.png")
                    static_plot = PlotFactory.plot_histogram_static(series, save_path)
                    interactive_plot = PlotFactory.plot_histogram_interactive(
                        series, annotations=annotations
                    )
                    plots[col] = {
                        "static": static_plot["path"],
                        "interactive": interactive_plot,
                    }
                    self._log(f"{col}: skewness computed", "debug")

            # Compute skewness for Pandas DataFrame
            else:
                numeric_df = (
                    df[numeric_cols]
                    if numeric_cols
                    else df.select_dtypes(include="number")
                )
                for col in numeric_df.columns:
                    series = numeric_df[col].dropna()
                    if series.empty:
                        self._log(f"{col} skipped: empty after dropna()", "debug")
                        continue
                    if series.nunique() == 1:
                        skew_val = 0.0
                        self._log(f"{col} skipped: constant values", "debug")
                    else:
                        skew_val = skew(series)
                    skewness[col] = float(skew_val)

                    annotations = [f"Skewness: {skew_val:.3f}"]
                    save_path = self.get_output_path(f"{col}_histogram.png")
                    static_plot = PlotFactory.plot_histogram_static(series, save_path)
                    interactive_plot = PlotFactory.plot_histogram_interactive(
                        series, annotations=annotations
                    )
                    plots[col] = {
                        "static": static_plot["path"],
                        "interactive": interactive_plot,
                    }
                    self._log(f"{col}: skewness computed", "debug")

            # Build TaskResult with visualization hints
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Computed skewness for {len(skewness)} numeric column(s)."
                    )
                },
                data=skewness,
                plots=plots,
                metadata={
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Distributions",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        numeric_cols + list(excluded.keys())
                    ),
                },
            )

            # Optional ML impact scoring based on skew severity
            if self.get_engine_param("enable_impact_scoring", True):
                for col, skew_val in skewness.items():
                    abs_skew = abs(skew_val)
                    if abs_skew <= 1:
                        continue  # low risk
                    score = 0.6 if abs_skew <= 2 else 0.8
                    tip = get_recommendation_tip(self.name, {"skew": skew_val})
                    self.set_ml_signals(
                        result=self.output,
                        score=score,
                        tags=["transform"],
                        recommendation=tip
                        or (
                            f"Column '{col}' has high skew (skew = {skew_val:.2f})."
                            " Consider transforming it."
                        ),
                    )
                    self.output.summary["column"] = col
                    break

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
