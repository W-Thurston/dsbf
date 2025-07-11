# dsbf/eda/tasks/detect_bimodal_distribution.py

from typing import Any, Dict

import numpy as np
from sklearn.mixture import GaussianMixture

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Detect Bimodal Distributions",
    description="Identifies columns with likely bimodal distributions.",
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="moderate",
    tags=["distribution", "outliers"],
    expected_semantic_types=["continuous"],
)
class DetectBimodalDistribution(BaseTask):
    """
    Uses Gaussian Mixture Models to flag numeric columns likely to follow
    a bimodal distribution, based on BIC improvement between 1 and 2 components.
    """

    def run(self) -> None:
        """
        Runs the bimodal detection task using Gaussian Mixture Models.
        Stores results as a TaskResult, including:
        - bimodal_flags: dict of column: bool
        - bic_scores: dict of column: {bic_1_component, bic_2_components}
        """

        try:

            # ctx = self.context
            df = self.input_data
            if is_polars(df):
                self._log(
                    "    Falling back to Pandas: sklearn GMM requires numeric arrays",
                    "debug",
                )
                df = df.to_pandas()  # sklearn requires numpy/pandas

            bic_threshold = float(self.get_task_param("bic_threshold") or 10.0)
            bimodal_flags: Dict[str, bool] = {}
            bic_scores: Dict[str, Dict[str, float]] = {}

            # Use semantic typing to select relevant columns
            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'continuous' column(s)", "debug"
            )

            numeric_df = df.select_dtypes(include=np.number)

            for col in numeric_df.columns:
                col_data = numeric_df[col].dropna().values.reshape(-1, 1)

                # Skip if not enough data points for GMM
                if col_data.shape[0] < 10:
                    continue

                if np.std(col_data) == 0 or np.unique(col_data).size < 2:
                    continue

                try:
                    gmm1 = GaussianMixture(n_components=1).fit(col_data)
                    gmm2 = GaussianMixture(n_components=2).fit(col_data)
                    bic1 = gmm1.bic(col_data)
                    bic2 = gmm2.bic(col_data)

                    bic_scores[col] = {
                        "bic_1_component": float(bic1),
                        "bic_2_components": float(bic2),
                    }
                    bimodal_flags[col] = bool((bic1 - bic2) > bic_threshold)
                except Exception as e:
                    self._log(f"    Failed on column {col}: {e}", "debug")
                    continue

            plots: dict[str, dict[str, Any]] = {}

            if self.context and self.context.output_dir:
                for col in bimodal_flags:
                    if col not in df.columns:
                        continue
                    series = df[col].dropna()
                    if series.empty:
                        continue

                    save_path = self.get_output_path(f"{col}_bimodal_hist.png")
                    static = PlotFactory.plot_histogram_static(series, save_path)

                    interactive = PlotFactory.plot_histogram_interactive(
                        series,
                        annotations=(
                            ["Possible bimodal distribution"]
                            if bimodal_flags[col]
                            else []
                        ),
                    )

                    plots[col] = {
                        "static": static["path"],
                        "interactive": interactive,
                    }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Flagged {sum(bimodal_flags.values())} "
                        f"column(s) as likely bimodal."
                    )
                },
                data={
                    "bimodal_flags": bimodal_flags,
                    "bic_scores": bic_scores,
                },
                plots=plots,
                metadata={
                    "bic_threshold": bic_threshold,
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Distributions",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys())
                    ),
                },
            )

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} — {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
