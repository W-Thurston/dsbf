# dsbf/eda/tasks/detect_target_drift.py

from typing import Any, Literal

import numpy as np
import polars as pl
from numpy import ndarray
from polars import DataFrame
from scipy.stats import chisquare, entropy, ks_2samp

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    name="detect_target_drift",
    display_name="Detect Target Drift",
    description=(
        "Detects distributional drift in the target column between current "
        "and reference datasets."
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
class DetectTargetDrift(BaseTask):
    """
    Detects distributional drift in the configured target column.

    Compares the target column's distribution between the current and reference
    datasets using type-appropriate metrics:

    - **Numeric targets**: PSI (Population Stability Index) and Kolmogorov-Smirnov
      test p-value.
    - **Categorical targets**: Chi-squared test p-value, Total Variation Distance
      (TVD), and entropy delta.

    Drift severity is classified as ``"none"``, ``"moderate"``, or
    ``"significant"`` based on configurable thresholds.

    The task returns a skipped result when any of the following are absent:
    - ``ctx.reference_data``
    - ``target`` parameter in task config
    - The target column in either dataset

    EDA guidance blurbs are emitted for moderate and significant drift.

    Configurable parameters (via config["tasks"]["detect_target_drift"]):
        target (str): Name of the target column to analyse.
        psi (float): PSI threshold for numeric drift severity. Default: 0.1
        ks_pvalue (float): KS test p-value threshold. Default: 0.05
        chi2_pvalue (float): Chi-squared p-value threshold for categorical.
            Default: 0.05
        entropy_delta (float): Entropy difference threshold. Default: 0.5
    """

    def run(self) -> None:
        """
        Execute target drift detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        ctx = self.context
        current_df = self.input_data
        reference_df: Any | None = getattr(ctx, "reference_data", None)

        matched_cols, excluded = self.get_columns_by_intent()
        self._log(f"    Processing {len(matched_cols)} column(s)", "debug")

        if reference_df is None:
            self.output = TaskResult(
                name=self.name,
                status="skipped",
                summary={"message": "[SKIPPED] No reference dataset provided."},
                data={},
                recommendations=[],
            )
            return

        target_col: str | None = self.get_task_param("target") or None
        if not target_col:
            self.output = TaskResult(
                name=self.name,
                status="skipped",
                summary={"message": "[SKIPPED] No target column specified in config."},
                data={},
                recommendations=[],
            )
            return

        if (
            target_col not in current_df.columns
            or target_col not in reference_df.columns
        ):
            self.output = TaskResult(
                name=self.name,
                status="skipped",
                summary={
                    "message": (
                        f"[SKIPPED] Target column '{target_col}' missing in "
                        "one of the datasets."
                    ),
                },
                data={},
                recommendations=[],
            )
            return

        current_series = current_df[target_col].drop_nulls()
        reference_series = reference_df[target_col].drop_nulls()

        try:
            if current_series.dtype.is_numeric():
                self.output = self._evaluate_numeric_drift(
                    current_series,
                    reference_series,
                    matched_cols,
                    excluded,
                )
            else:
                self.output = self._evaluate_categorical_drift(
                    current_series,
                    reference_series,
                    matched_cols,
                    excluded,
                )
        except Exception as e:  # noqa: BLE001
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _evaluate_numeric_drift(
        self,
        current: pl.Series,
        reference: pl.Series,
        matched_cols: list[str],
        excluded: dict,
    ) -> TaskResult:
        """
        Compute PSI and KS-test for a numeric target column.

        Args:
            current: Current dataset target Series (non-null values).
            reference: Reference dataset target Series (non-null values).
            matched_cols: Columns matched by semantic type.
            excluded: Columns excluded by semantic type.

        Returns:
            TaskResult with PSI, KS p-value, and drift severity.

        """
        psi_threshold = float(self.get_task_param("psi") or 0.1)
        ks_threshold = float(self.get_task_param("ks_pvalue") or 0.05)

        bins: ndarray = np.histogram_bin_edges(
            np.concatenate([current.to_numpy(), reference.to_numpy()]),
            bins=10,
        )
        current_counts, _ = np.histogram(current.to_numpy(), bins=bins)
        reference_counts, _ = np.histogram(reference.to_numpy(), bins=bins)

        current_pct: ndarray = np.where(
            current_counts == 0,
            1e-6,
            current_counts / current_counts.sum(),
        )
        reference_pct: ndarray = np.where(
            reference_counts == 0,
            1e-6,
            reference_counts / reference_counts.sum(),
        )

        psi = float(
            np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)),
        )
        _, ks_p = ks_2samp(current.to_numpy(), reference.to_numpy())

        drift_severity: Literal["moderate", "none", "significant"] = (
            "significant"
            if psi >= 0.25 or ks_p < ks_threshold / 2  # noqa: PLR2004
            else ("moderate" if psi >= psi_threshold or ks_p < ks_threshold else "none")
        )

        recommendations: list[str] = []
        if drift_severity in ("moderate", "significant"):
            recommendations.append(
                "Consider retraining or validating your model due to target drift.",
            )

        result = TaskResult(
            name=self.name,
            status="success",
            summary={
                "message": (
                    f"Target drift: {drift_severity.upper()} "
                    f"(PSI={psi:.3f}, KS p={ks_p:.3f})"
                ),
            },
            data={
                "target_type": "numerical",
                "psi": float(psi),
                "ks_pvalue": round(float(ks_p), 4),
                "drift_rating": drift_severity,
            },
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

        if drift_severity != "none":
            self._attach_guidance_numeric(result, psi, float(ks_p), drift_severity)

        if (
            self.get_engine_param("enable_impact_scoring", True)
            and drift_severity != "none"
        ):
            tip: str | None = get_recommendation_tip(
                self.name,
                {"drift_rating": drift_severity},
            )
            self.set_ml_signals(
                result=result,
                score=0.8,
                tags=["monitor"],
                recommendation=tip
                or (
                    f"Target drift detected (severity: {drift_severity}). "
                    "Consider retraining or validating model performance."
                ),
            )
            result.summary["column"] = self.get_task_param("target")

        return result

    def _evaluate_categorical_drift(
        self,
        current: pl.Series,
        reference: pl.Series,
        matched_cols: list[str],
        excluded: dict,
    ) -> TaskResult:
        """
        Compute TVD, chi-squared test, and entropy delta for a categorical target.

        Args:
            current: Current dataset target Series (non-null values).
            reference: Reference dataset target Series (non-null values).
            matched_cols: Columns matched by semantic type.
            excluded: Columns excluded by semantic type.

        Returns:
            TaskResult with TVD, chi-squared p-value, entropy delta, and severity.

        """
        chi2_threshold = float(self.get_task_param("chi2_pvalue") or 0.05)
        entropy_threshold = float(self.get_task_param("entropy_delta") or 0.5)

        current_counts: DataFrame = current.value_counts().sort(current.name)
        reference_counts: DataFrame = reference.value_counts().sort(reference.name)

        categories: list[Any] = sorted(
            set(current_counts[current.name].to_list())
            | set(reference_counts[reference.name].to_list()),
        )
        current_freq: dict[Any, int] = dict.fromkeys(categories, 0)
        reference_freq: dict[Any, int] = dict.fromkeys(categories, 0)

        for row in current_counts.iter_rows():
            current_freq[row[0]] = row[1]
        for row in reference_counts.iter_rows():
            reference_freq[row[0]] = row[1]

        observed: ndarray = np.array([current_freq[c] for c in categories])
        expected: ndarray = np.array([reference_freq[c] for c in categories])

        total_obs = observed.sum()
        total_exp = expected.sum()

        if total_obs == 0 or total_exp == 0:
            raise ValueError("Target column has no data in one of the datasets.")

        observed_pct = observed / total_obs
        expected_pct = expected / total_exp

        tvd = float(0.5 * np.sum(np.abs(observed_pct - expected_pct)))
        _, chi2_p = chisquare(f_obs=observed, f_exp=expected)

        entropy_current = float(entropy(observed_pct + 1e-6))
        entropy_reference = float(entropy(expected_pct + 1e-6))
        entropy_delta: float = abs(entropy_current - entropy_reference)

        drift_severity: Literal["moderate", "none", "significant"] = (
            "significant"
            if chi2_p < chi2_threshold / 2 or entropy_delta > 2 * entropy_threshold
            else (
                "moderate"
                if chi2_p < chi2_threshold or entropy_delta > entropy_threshold
                else "none"
            )
        )

        recommendations: list[str] = []
        if drift_severity in ("moderate", "significant"):
            recommendations.append(
                "Class distribution drift detected. "
                "Retrain or monitor model performance.",
            )

        result = TaskResult(
            name=self.name,
            status="success",
            summary={
                "message": (
                    f"Target drift: {drift_severity.upper()} "
                    f"(TVD={tvd:.3f}, Chi² p={chi2_p:.3f})"
                ),
            },
            data={
                "target_type": "categorical",
                "tvd": float(tvd),
                "chi2_pvalue": round(float(chi2_p), 4),
                "entropy_delta": float(entropy_delta),
                "drift_rating": drift_severity,
            },
            recommendations=recommendations,
            metadata={
                "suggested_viz_type": "bar",
                "recommended_section": "Comparison",
                "display_priority": "high",
                "excluded_columns": excluded,
                "column_types": self.get_column_type_info(
                    matched_cols + list(excluded.keys()),
                ),
            },
        )

        if drift_severity != "none":
            self._attach_guidance_categorical(
                result,
                tvd,
                float(chi2_p),
                entropy_delta,
                drift_severity,
            )

        if (
            self.get_engine_param("enable_impact_scoring", True)
            and drift_severity != "none"
        ):
            tip: str | None = get_recommendation_tip(
                self.name,
                {"drift_rating": drift_severity},
            )
            self.set_ml_signals(
                result=result,
                score=0.8,
                tags=["monitor"],
                recommendation=tip
                or (
                    f"Target drift detected (severity: {drift_severity}). "
                    "Consider retraining or validating model performance."
                ),
            )
            result.summary["column"] = self.get_task_param("target")

        return result

    def _attach_guidance_numeric(
        self,
        result: TaskResult,
        psi: float,
        ks_p: float,
        severity: str,
    ) -> None:
        """
        Attach EDA guidance for numeric target drift.

        Args:
            result: TaskResult to attach guidance to.
            psi: PSI value.
            ks_p: KS test p-value.
            severity: ``"moderate"`` or ``"significant"``.

        """
        target_col: Literal["target"] | Any = self.get_task_param("target") or "target"
        level: Literal["error", "warn"] = "warn" if severity == "moderate" else "error"

        eda_body: str = (
            f"The numeric target '{target_col}' shows {severity} distributional drift "
            f"between the reference and current datasets (PSI = {psi:.4f}, "
            f"KS p-value = {ks_p:.4f}). The distribution of target values has "
            f"shifted substantially. This may indicate concept drift, a change in "
            f"the population being served, or a data pipeline issue. Model performance "
            f"predictions based on historical validation may no longer be reliable."
        )

        self.add_guidance(
            result=result,
            column=target_col,
            phase="eda",
            level=level,
            title=f"Target Drift Detected - {severity.title()} (PSI={psi:.3f})",
            body=eda_body.strip(),
            actions=[],
            metric={
                "psi": round(psi, 4),
                "ks_pvalue": round(ks_p, 4),
                "severity": severity,
            },
        )

    def _attach_guidance_categorical(
        self,
        result: TaskResult,
        tvd: float,
        chi2_p: float,
        entropy_delta: float,
        severity: str,
    ) -> None:
        """
        Attach EDA guidance for categorical target drift.

        Args:
            result: TaskResult to attach guidance to.
            tvd: Total Variation Distance.
            chi2_p: Chi-squared test p-value.
            entropy_delta: Absolute entropy difference.
            severity: ``"moderate"`` or ``"significant"``.

        """
        target_col: Literal["target"] | Any = self.get_task_param("target") or "target"
        level: Literal["error", "warn"] = "warn" if severity == "moderate" else "error"

        eda_body: str = (
            f"The categorical target '{target_col}' shows {severity} class "
            f"distribution drift between the reference and current datasets "
            f"(TVD = {tvd:.4f}, Chi² p = {chi2_p:.4f}, entropy delta = "
            f"{entropy_delta:.4f}). The class proportions have shifted. If the model "
            f"was trained on the reference distribution, its decision boundaries and "
            f"threshold calibration may no longer match the current population."
        )

        self.add_guidance(
            result=result,
            column=target_col,
            phase="eda",
            level=level,
            title=(
                f"Target Class Distribution Drift - {severity.title()} (TVD={tvd:.3f})"
            ),
            body=eda_body.strip(),
            actions=[],
            metric={
                "tvd": round(tvd, 4),
                "chi2_pvalue": round(chi2_p, 4),
                "entropy_delta": round(entropy_delta, 4),
                "severity": severity,
            },
        )
