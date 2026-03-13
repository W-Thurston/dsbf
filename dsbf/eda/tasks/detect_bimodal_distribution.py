# dsbf/eda/tasks/detect_bimodal_distribution.py

from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


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
            # Relative threshold: BIC scales with N, so an absolute threshold
            # produces false positives on large datasets. Require the 2-component
            # model to improve BIC by at least this fraction of the 1-component BIC.
            # Default 1% filters noise while preserving genuine bimodal signal.
            relative_bic_threshold = float(
                self.get_task_param("relative_bic_threshold") or 0.01
            )
            bimodal_flags: dict[str, bool] = {}
            bic_scores: dict[str, dict[str, float]] = {}

            # Use semantic typing to select relevant columns
            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'continuous' column(s)",
                "debug",
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
                    # Guard: highly skewed distributions look bimodal to GMM but are not
                    # A skewness guard above 2.0 filters right-skewed(e.g. fare, income)
                    # and left-skewed columns before running the expensive GMM test.
                    skew_threshold = float(self.get_task_param("skewness_guard") or 2.0)
                    col_skewness = float(numeric_df[col].skew())
                    if abs(col_skewness) > skew_threshold:
                        bic_scores[col] = {
                            "bic_1_component": None,
                            "bic_2_components": None,
                            "delta": None,
                            "relative_improvement": None,
                            "skipped": "high_skewness",
                            "skewness": col_skewness,
                        }
                        bimodal_flags[col] = False
                        continue

                    gmm1: GaussianMixture = GaussianMixture(
                        n_components=1,
                        random_state=42,
                    ).fit(
                        col_data,
                    )
                    gmm2: GaussianMixture = GaussianMixture(
                        n_components=2,
                        random_state=42,
                    ).fit(
                        col_data,
                    )
                    bic1 = gmm1.bic(col_data)
                    bic2 = gmm2.bic(col_data)

                    delta = bic1 - bic2
                    bic_scores[col] = {
                        "bic_1_component": float(bic1),
                        "bic_2_components": float(bic2),
                        "delta": float(delta),
                        "relative_improvement": (
                            float(delta / abs(bic1)) if abs(bic1) > 0 else 0.0
                        ),
                    }
                    # Use relative improvement so the threshold stays meaningful
                    # regardless of dataset size. Falls back to absolute check if
                    # bic1 is zero (shouldn't occur in practice).
                    rel_improvement: float = delta / abs(bic1) if abs(bic1) > 0 else 0.0
                    bimodal_flags[col] = bool(
                        delta > bic_threshold
                        and rel_improvement > relative_bic_threshold,
                    )
                except Exception as e:
                    self._log(f"    Failed on column {col}: {e}", "debug")
                    continue

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Flagged {sum(bimodal_flags.values())} "
                        f"column(s) as likely bimodal."
                    ),
                },
                data={
                    "bimodal_flags": bimodal_flags,
                    "bic_scores": bic_scores,
                },
                plots={},
                metadata={
                    "bic_threshold": bic_threshold,
                    "relative_bic_threshold": relative_bic_threshold,
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Distributions",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, is_bimodal in bimodal_flags.items():
                if is_bimodal:
                    self._attach_guidance(col, bic_scores.get(col, {}))

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, bic_data: dict[str, Any]) -> None:
        """Generate EDA + ML guidance for a flagged bimodal column."""
        delta: Any | None = bic_data.get("delta")
        rel_improvement: Any | None = bic_data.get("relative_improvement")

        # Strength label based on relative BIC improvement
        # rel_improvement > 0.05 is a strong two-peak signal
        if rel_improvement is not None and rel_improvement > 0.05:
            strength = "strongly"
            level = "warn"
        else:
            strength = "likely"
            level = "info"

        delta_str: str = f"{delta:.1f}" if delta is not None else "N/A"
        rel_str: str = (
            f"{rel_improvement:.1%}" if rel_improvement is not None else "N/A"
        )

        eda_body: str = (
            f"{col} {strength} has a bimodal distribution - a two-component "
            f"Gaussian model fits the data {rel_str} better than a single "
            f"Gaussian (BIC improvement: {delta_str}). This means the values "
            f"cluster around two distinct centres rather than one. Bimodality "
            f"often signals that the column is a mixture of two underlying "
            f"populations - for example, two seasons, two measurement instruments, "
            f"two demographic groups, or two distinct processes generating the data. "
            f"Examine the histogram and consider whether a known categorical split "
            f"(e.g. by group or time period) explains the two peaks."
        )

        ml_body: str = (
            f"{col} has a bimodal distribution (BIC improvement: {delta_str}, "
            f"{rel_str} relative). A single Gaussian assumption will misfit this "
            f"column. Models sensitive to distributional shape - linear regression, "
            f"LDA, Gaussian Naive Bayes - will be affected. Tree-based models "
            f"(Random Forest, Gradient Boosting) handle bimodality natively by "
            f"splitting on thresholds. If the source of bimodality is known "
            f"(e.g. a group variable), consider adding that variable or an "
            f"interaction term. If unknown, a cluster membership indicator "
            f"derived from the two-component GMM may improve model performance."
        )

        metric: dict[str | None] = {
            "bic_delta": round(delta, 2) if delta is not None else None,
            "relative_improvement": (
                round(rel_improvement, 4) if rel_improvement is not None else None
            ),
            "bic_1_component": (
                round(bic_data.get("bic_1_component", 0), 2)
                if bic_data.get("bic_1_component") is not None
                else None
            ),
            "bic_2_components": (
                round(bic_data.get("bic_2_components", 0), 2)
                if bic_data.get("bic_2_components") is not None
                else None
            ),
        }

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=f"Bimodal Distribution ({strength.title()} - {rel_str} "
            "BIC improvement)",
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=level,
            title="Non-Gaussian Shape - Two Peaks Detected",
            body=ml_body.strip(),
            actions=[
                {
                    "action": "segment",
                    "column": col,
                    "detail": "Split by known group variable if "
                    "the source of bimodality is understood",
                },
                {
                    "action": "add_feature",
                    "method": "gmm_cluster_indicator",
                    "column": col,
                    "detail": "Derive a binary cluster membership "
                    "feature from the 2-component GMM",
                },
                {
                    "action": "use_tree_model",
                    "detail": "Tree-based models (RF, GBM) "
                    "handle bimodality natively without transformation",
                },
            ],
            metric=metric,
        )
