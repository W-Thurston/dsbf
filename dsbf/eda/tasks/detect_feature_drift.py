# dsbf/eda/tasks/detect_feature_drift.py

from typing import Any, Literal

import numpy as np
import polars as pl
from numpy import ndarray
from pandas import DataFrame
from polars import Series
from scipy.stats import chi2_contingency, ks_2samp

from dsbf.core.base_task import BaseTask
from dsbf.core.context import AnalysisContext
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_text_polars
from dsbf.utils.reco_engine import get_recommendation_tip


def compute_psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index between a reference and current distribution.

    PSI measures how much a distribution has shifted. Industry thresholds:
    - PSI < 0.1: negligible drift
    - 0.1 ≤ PSI < 0.2: moderate drift — worth monitoring
    - PSI ≥ 0.2: significant drift — model may need retraining

    Args:
        ref: Reference distribution as a 1-D numpy array.
        cur: Current distribution as a 1-D numpy array.
        bins: Number of histogram bins. Default: 10

    Returns:
        PSI value (≥ 0.0). Higher values indicate greater distributional shift.

    """
    combined_min = min(ref.min(), cur.min())
    combined_max = max(ref.max(), cur.max())
    ref_percents, _ = np.histogram(
        ref,
        bins=bins,
        range=(combined_min, combined_max),
        density=True,
    )
    cur_percents, _ = np.histogram(
        cur,
        bins=bins,
        range=(combined_min, combined_max),
        density=True,
    )
    # Replace zeros to avoid log(0) — small epsilon preserves direction of change.
    ref_percents: ndarray = np.where(ref_percents == 0, 1e-6, ref_percents)
    cur_percents: ndarray = np.where(cur_percents == 0, 1e-6, cur_percents)
    return float(
        np.sum((ref_percents - cur_percents) * np.log(ref_percents / cur_percents)),
    )


def get_severity(value: float, threshold: float) -> str:
    """
    Classify a drift metric value into a severity tier.

    Args:
        value: Drift metric value (PSI or TVD).
        threshold: The baseline threshold for the metric.

    Returns:
        ``"low"`` if below threshold, ``"moderate"`` if below 2x threshold,
        ``"high"`` if at or above 2x threshold.

    """
    if value < threshold:
        return "low"
    if value < 2 * threshold:
        return "moderate"
    return "high"


@register_task(
    name="detect_feature_drift",
    display_name="Detect Feature Drift",
    description=(
        "Detects distributional drift between current and reference datasets "
        "for shared columns."
    ),
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="slow",
    phase="eda",
    tags=["drift", "comparison"],
    expected_semantic_types=["any"],
)
class DetectFeatureDrift(BaseTask):
    """
    Detects distributional drift between the current and reference datasets.

    For each column shared between the current and reference DataFrames:

    - **Numeric columns**: PSI (Population Stability Index) and a
      Kolmogorov-Smirnov test p-value are computed.
    - **Categorical columns**: Total Variation Distance (TVD) and a
      chi-squared independence test p-value are computed.

    Severity is classified as ``"low"``, ``"moderate"``, or ``"high"`` based on
    configurable thresholds. Guidance blurbs are emitted for columns with
    ``"high"`` severity drift.

    If no reference dataset is available in ``ctx.reference_data``, the task
    returns a skipped result.

    Requires a Polars-native reference DataFrame. Pandas reference DataFrames
    are not currently supported.

    Configurable parameters (via config["tasks"]["detect_feature_drift"]):
        psi (float): PSI threshold for numeric drift severity. Default: 0.1
        tvd (float): TVD threshold for categorical drift severity. Default: 0.2
    """

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """
        Execute feature drift detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            ctx: AnalysisContext | None = self.context
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} column(s)",
                "debug",
            )

            reference: pl.DataFrame | None = getattr(ctx, "reference_data", None)
            if reference is None:
                self.output = TaskResult(
                    name=self.name,
                    status="skipped",
                    summary={"message": "Reference dataset not provided in context."},
                )
                return

            shared_cols: list[str] = [
                col for col in df.columns if col in reference.columns
            ]
            if not shared_cols:
                self.output = TaskResult(
                    name=self.name,
                    status="skipped",
                    summary={
                        "message": (
                            "No shared columns between current and reference datasets."
                        ),
                    },
                )
                return

            psi_threshold = float(self.get_task_param("psi") or 0.1)
            tvd_threshold = float(self.get_task_param("tvd") or 0.2)

            drift_results: dict = {}
            numeric_cols: list[str] = []
            categorical_cols: list[str] = []

            for col in shared_cols:
                try:
                    current_col = df.get_column(col)
                    reference_col: Series = reference.get_column(col)

                    if (
                        hasattr(current_col.dtype, "is_numeric")
                        and current_col.dtype.is_numeric()
                        and hasattr(reference_col.dtype, "is_numeric")
                        and reference_col.dtype.is_numeric()
                    ):
                        numeric_cols.append(col)

                        cur_np = current_col.drop_nulls().to_numpy()
                        ref_np: ndarray[Any, Any] = (
                            reference_col.drop_nulls().to_numpy()
                        )

                        if len(cur_np) == 0 or len(ref_np) == 0:
                            drift_results[col] = {
                                "type": "numerical",
                                "error": "Empty array after null removal",
                            }
                            continue

                        psi: float = compute_psi(ref_np, cur_np)
                        _, ks_p = ks_2samp(ref_np, cur_np)
                        severity: str = get_severity(psi, psi_threshold)

                        drift_results[col] = {
                            "type": "numerical",
                            "psi": round(psi, 4),
                            "ks_pvalue": round(float(ks_p), 4),
                            "severity": severity,
                        }

                    elif is_text_polars(current_col) and is_text_polars(reference_col):
                        categorical_cols.append(col)

                        cur_vals = current_col.drop_nulls().cast(str).value_counts()
                        ref_vals: DataFrame = (
                            reference_col.drop_nulls().cast(str).value_counts()
                        )

                        col_name, count_name = cur_vals.columns
                        cur_dict: dict = {
                            row[col_name]: row[count_name]
                            for row in cur_vals.iter_rows(named=True)
                        }
                        ref_dict: dict = {
                            row[col_name]: row[count_name]
                            for row in ref_vals.iter_rows(named=True)
                        }

                        all_keys: set = set(cur_dict.keys()) | set(ref_dict.keys())
                        total_cur: int = sum(cur_dict.values())
                        total_ref: int = sum(ref_dict.values())

                        tvd: float = 0.5 * sum(
                            abs(
                                (cur_dict.get(k, 0) / total_cur)
                                - (ref_dict.get(k, 0) / total_ref),
                            )
                            for k in all_keys
                        )

                        table: list[list] = [
                            [cur_dict.get(k, 0) for k in all_keys],
                            [ref_dict.get(k, 0) for k in all_keys],
                        ]
                        _, chi2_p, _, _ = chi2_contingency(table)
                        severity = get_severity(tvd, tvd_threshold)

                        drift_results[col] = {
                            "type": "categorical",
                            "tvd": round(tvd, 4),
                            "chi2_pvalue": round(float(chi2_p), 4),
                            "severity": severity,
                        }
                    else:
                        drift_results[col] = {
                            "type": "unsupported",
                            "error": (
                                f"'{col}' is neither numeric nor string — skipped."
                            ),
                        }

                except Exception as e:  # noqa: BLE001
                    drift_results[col] = {"type": "unknown", "error": str(e)}

            high_drift_cols: list = [
                col
                for col, res in drift_results.items()
                if res.get("severity") == "high"
            ]

            recommendations: list[str] = []
            if high_drift_cols:
                recommendations.append(
                    f"High drift detected in columns: {high_drift_cols}. "
                    "Consider reviewing the data pipeline or retraining the model.",
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "total_columns_evaluated": len(shared_cols),
                    "numeric_columns_checked": len(numeric_cols),
                    "categorical_columns_checked": len(categorical_cols),
                    "high_drift_columns": high_drift_cols,
                },
                data=drift_results,
                recommendations=recommendations,
                metadata={
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Comparison",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col in high_drift_cols:
                self._attach_guidance(
                    col,
                    drift_results[col],
                    psi_threshold,
                    tvd_threshold,
                )

            # ML impact scoring
            if self.get_engine_param("enable_impact_scoring", True) and high_drift_cols:
                top_col = high_drift_cols[0]
                drift_info = drift_results[top_col]
                metric_name: Literal["psi", "tvd"] = (
                    "psi" if drift_info.get("type") == "numerical" else "tvd"
                )
                value = drift_info.get(metric_name)
                tip: str | None = get_recommendation_tip(self.name, {"psi": value})
                self.set_ml_signals(
                    result=self.output,
                    score=0.7,
                    tags=["monitor"],
                    recommendation=tip
                    or (
                        f"Column '{top_col}' shows high drift "
                        f"({metric_name} = {value}). This may indicate a shift "
                        "in data distribution — monitor closely or retrain."
                    ),
                )
                self.output.summary["column"] = top_col

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(
        self,
        col: str,
        drift_info: dict,
        psi_threshold: float,
        tvd_threshold: float,
    ) -> None:
        """
        Generate EDA guidance for a high-drift column.

        Args:
            col: Column name.
            drift_info: Drift result dict for this column from ``drift_results``.
            psi_threshold: Configured PSI threshold for severity classification.
            tvd_threshold: Configured TVD threshold for severity classification.

        """
        col_type = drift_info.get("type", "unknown")

        if col_type == "numerical":
            psi = drift_info.get("psi", 0.0)
            ks_p = drift_info.get("ks_pvalue")
            ks_str: str = f", KS p-value: {ks_p:.4f}" if ks_p is not None else ""
            metric_str: str = f"PSI = {psi:.4f}{ks_str}"
            threshold_str: str = f"PSI threshold: {psi_threshold}"
            metric = {"psi": psi, "ks_pvalue": ks_p, "psi_threshold": psi_threshold}
        else:
            tvd = drift_info.get("tvd", 0.0)
            chi2_p = drift_info.get("chi2_pvalue")
            chi2_str = f", chi² p-value: {chi2_p:.4f}" if chi2_p is not None else ""
            metric_str = f"TVD = {tvd:.4f}{chi2_str}"
            threshold_str = f"TVD threshold: {tvd_threshold}"
            metric = {"tvd": tvd, "chi2_pvalue": chi2_p, "tvd_threshold": tvd_threshold}

        eda_body = (
            f"'{col}' shows high distributional drift between the reference and "
            f"current datasets ({metric_str}; {threshold_str}). The distribution "
            f"of values has shifted substantially, which may indicate a change in "
            f"the data collection process, a schema migration, a seasonal effect, "
            f"or a genuine shift in the underlying population. Inspect the "
            f"distributions side by side and review the data pipeline for this "
            f"column before drawing analytical conclusions."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="warn",
            title=f"High Distributional Drift ({metric_str})",
            body=eda_body.strip(),
            actions=[],
            metric=metric,
        )
