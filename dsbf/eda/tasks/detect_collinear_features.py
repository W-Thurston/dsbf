# dsbf/eda/tasks/detect_collinear_features.py

from typing import Dict, List

import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    make_failure_result,
)
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    display_name="Detect Collinear Features",
    description="Detects highly collinear features that may cause multicollinearity.",
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="modeling",
    domain="core",
    runtime_estimate="slow",
    tags=["multicollinearity", "numeric"],
    expected_semantic_types=["continuous"],
)
class DetectCollinearFeatures(BaseTask):
    def run(self) -> None:
        try:
            df = self.input_data
            flags = self.ensure_reliability_flags()

            vif_threshold = float(self.get_task_param("vif_threshold") or 10.0)

            if is_polars(df):
                self._log(
                    "    Falling back to Pandas: VIF calculation requires NumPy arrays",
                    "debug",
                )
                df = df.to_pandas()

            # Use semantic typing to select relevant columns
            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'continuous' column(s)", "debug"
            )
            numeric_df = df.select_dtypes(include=np.number).dropna()

            if numeric_df.shape[1] < 2:
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary={"message": "Not enough numeric features to compute VIF."},
                    data={"vif_scores": {}, "collinear_columns": []},
                    metadata={"vif_threshold": vif_threshold},
                )
                return

            vif_scores: Dict[str, float] = {}
            for i in range(numeric_df.shape[1]):
                col = numeric_df.columns[i]
                vif_val = variance_inflation_factor(numeric_df.values, i)
                vif_scores[col] = float(vif_val)

            collinear_columns: List[str] = [
                col for col, vif in vif_scores.items() if vif > vif_threshold
            ]

            result = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Flagged {len(collinear_columns)} "
                        f"column(s) with VIF > {vif_threshold}."
                    )
                },
                data={
                    "vif_scores": vif_scores,
                    "collinear_columns": collinear_columns,
                },
                metadata={
                    "vif_threshold": vif_threshold,
                    "suggested_viz_type": "heatmap",
                    "recommended_section": "Multicollinearity",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys())
                    ),
                },
            )

            # Reliability warnings
            if flags["low_row_count"]:
                add_reliability_warning(
                    result,
                    level="heuristic_caution",
                    code="vif_low_n",
                    description=(
                        "VIF values may be unstable when"
                        " sample size is small (N < 30)."
                    ),
                    recommendation=(
                        "Consider bootstrapping or collecting"
                        " more data before interpreting VIF."
                    ),
                )
            if flags["zero_variance_cols"]:
                add_reliability_warning(
                    result,
                    level="strong_warning",
                    code="vif_zero_variance",
                    description=(
                        "Some features have near-zero variance,"
                        " which can distort VIF calculations."
                    ),
                    recommendation=(
                        "Drop or transform zero-variance features"
                        " before running VIF."
                    ),
                )

            # Apply ML scoring to self.output
            if (
                self.get_engine_param("enable_impact_scoring", True)
                and collinear_columns
            ):
                top_col = collinear_columns[0]
                top_vif = vif_scores[top_col]
                score = 0.75 if top_vif < 15 else 0.85
                tip = get_recommendation_tip(self.name, {"vif": top_vif})
                self.set_ml_signals(
                    result=result,
                    score=score,
                    tags=["drop", "transform"],
                    recommendation=tip
                    or (
                        f"Column '{top_col}' has high multicollinearity"
                        f" (VIF = {top_vif:.2f}). "
                        "Consider dropping this feature or applying regularization/PCA."
                    ),
                )
                result.summary["column"] = top_col

            try:
                save_path = self.get_output_path("correlation_heatmap.png")
                static = PlotFactory.plot_correlation_static(
                    df, save_path=save_path, title="Correlation Matrix"
                )
                if vif_scores:
                    top_vif_col = max(vif_scores.items(), key=lambda kv: kv[1])[0]
                    top_vif_val = vif_scores[top_vif_col]
                    annotation_str = f"Top VIF: {top_vif_col} ({top_vif_val:.2f})"
                else:
                    annotation_str = "No numeric features"

                annotations = [annotation_str]
                interactive = PlotFactory.plot_correlation_interactive(
                    df,
                    title="Correlation Matrix",
                    annotations=annotations,
                )
                result.plots = {
                    "correlation_matrix": {
                        "static": static["path"],
                        "interactive": interactive,
                    }
                }
            except Exception as e:
                self._log(
                    f"    [PlotFactory] Skipped correlation matrix plot: {e}",
                    level="debug",
                )

            self.output = result

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} — {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
