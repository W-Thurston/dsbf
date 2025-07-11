# dsbf/eda/tasks/detect_outliers.py

from typing import Any, Dict, List

import numpy as np

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    display_name="Detect Outliers",
    description="Uses statistical heuristics to flag outlier values.",
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="moderate",
    tags=["outliers", "numeric"],
    expected_semantic_types=["continuous"],
)
class DetectOutliers(BaseTask):
    """
    Detects numeric outliers using the IQR method. Flags columns exceeding
    a proportion threshold of outliers.
    """

    def run(self) -> None:
        try:

            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_col)} 'continuous' column(s)", "debug"
            )

            method = str(self.get_task_param("method") or "iqr")
            flag_threshold = float(self.get_task_param("flag_threshold") or 0.01)

            if is_polars(df):
                df = df.to_pandas()

            if not hasattr(df, "shape"):
                raise ValueError("Input is not a valid dataframe.")

            n_rows = df.shape[0]
            outlier_counts: Dict[str, int] = {}
            outlier_flags: Dict[str, bool] = {}
            outlier_rows: Dict[str, List[int]] = {}

            plots: Dict[str, Dict[str, Any]] = {}

            numeric_df = df.select_dtypes(include=[np.number])

            for col in numeric_df.columns:
                series = numeric_df[col].dropna()

                # Skip plotting + computation if no valid values remain
                if series.empty:
                    self._log(f"    {col} skipped: empty after dropna()", "debug")
                    continue

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_mask = (series < lower) | (series > upper)
                indices = series[outlier_mask].index.tolist()

                outlier_counts[col] = len(indices)
                outlier_rows[col] = indices
                outlier_flags[col] = len(indices) > flag_threshold * n_rows

                # Plot boxplot
                save_path = self.get_output_path(f"{col}_boxplot.png")
                static = PlotFactory.plot_boxplot_static(series, save_path)["path"]
                annotations = [
                    f"IQR: {iqr:.3f}",
                    f"Lower bound: {lower:.3f}",
                    f"Upper bound: {upper:.3f}",
                    f"Outliers detected: {outlier_mask.sum()}",
                ]
                interactive = PlotFactory.plot_boxplot_interactive(
                    series, annotations=annotations
                )

                plots[col] = {"static": static, "interactive": interactive}

            flagged_cols = [col for col, flagged in outlier_flags.items() if flagged]

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (f"Detected outliers in {len(flagged_cols)} column(s).")
                },
                data={
                    "outlier_counts": outlier_counts,
                    "outlier_flags": outlier_flags,
                    "outlier_rows": outlier_rows,
                },
                metadata={
                    "method": method,
                    "threshold_pct": flag_threshold,
                    "total_rows": n_rows,
                    "suggested_viz_type": "boxplot",
                    "recommended_section": "Outliers",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
                plots=plots,
            )

            # Apply ML scoring to self.output
            if self.get_engine_param("enable_impact_scoring", True) and flagged_cols:
                col = flagged_cols[0]
                n_outliers = outlier_counts[col]
                tip = get_recommendation_tip(self.name, {"n_outliers": n_outliers})
                self.set_ml_signals(
                    result=self.output,
                    score=0.7,
                    tags=["monitor", "transform"],
                    recommendation=tip
                    or (
                        f"Column '{col}' contains {n_outliers} statistical outliers. "
                        "Consider log-transforming, winsorizing,"
                        " or using robust models."
                    ),
                )
                self.output.summary["column"] = col

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} — {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
