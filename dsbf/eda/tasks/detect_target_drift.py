import numpy as np
import polars as pl
from scipy.stats import chisquare, entropy, ks_2samp

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult


@register_task(
    name="detect_target_drift",
    display_name="Detect Target Drift",
    description=(
        "Detects distributional drift between current and "
        "reference datasets for shared columns."
    ),
    depends_on=["infer_types"],
    profiling_depth="full",
    stage="cleaned",
    tags=["drift", "comparison"],
)
class DetectTargetDrift(BaseTask):
    """
    Detects distributional drift in the target column between the current and
        reference datasets.

    Supports both categorical and numerical targets with appropriate metrics:
      - Numerical: PSI, KS-test, mean/variance delta
      - Categorical: Chi-squared test, TVD (Total Variation Distance), entropy delta

    Requires that a reference dataset and target column are defined in the config.
    Skips execution gracefully if either is missing.
    """

    def run(self) -> None:
        ctx = self.context
        current_df = self.input_data
        reference_df = getattr(ctx, "reference_data", None)

        if reference_df is None:
            self.output = TaskResult(
                name=self.name,
                status="skipped",
                summary={"message": ("[SKIPPED] No reference dataset provided.")},
                data={},
                recommendations=[],
            )
            return

        target_col = self.get_task_param("target") or None
        if not target_col:
            self.output = TaskResult(
                status="skipped",
                name=self.name,
                summary={
                    "message": ("[SKIPPED] No target column specified in config.")
                },
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
                        f"[SKIPPED] Target column '{target_col}'"
                        " missing in one of the datasets."
                    )
                },
                data={},
                recommendations=[],
            )
            return

        current_series = current_df[target_col].drop_nulls()
        reference_series = reference_df[target_col].drop_nulls()

        try:
            if current_series.dtype.is_numeric():
                result = self._evaluate_numeric_drift(current_series, reference_series)
            else:
                result = self._evaluate_categorical_drift(
                    current_series, reference_series
                )
        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary={
                    "message": (
                        f"[ERROR] Failed to compute drift for target column: {e}"
                    )
                },
                data={},
                recommendations=[],
            )
            return

        self.output = result

    def _evaluate_numeric_drift(
        self, current: pl.Series, reference: pl.Series
    ) -> TaskResult:
        psi_threshold = float(self.get_task_param("psi") or 0.1)
        ks_threshold = float(self.get_task_param("ks_pvalue") or 0.05)

        # Compute PSI
        bins = np.histogram_bin_edges(
            np.concatenate([current.to_numpy(), reference.to_numpy()]), bins=10
        )
        current_counts, _ = np.histogram(current.to_numpy(), bins=bins)
        reference_counts, _ = np.histogram(reference.to_numpy(), bins=bins)

        current_pct = np.where(
            current_counts == 0, 1e-6, current_counts / current_counts.sum()
        )
        reference_pct = np.where(
            reference_counts == 0, 1e-6, reference_counts / reference_counts.sum()
        )

        psi = np.sum(
            (current_pct - reference_pct) * np.log(current_pct / reference_pct)
        )

        # KS-test
        ks_stat, ks_p = ks_2samp(current.to_numpy(), reference.to_numpy())

        drift_severity = (
            "significant"
            if psi >= 0.25 or ks_p < ks_threshold / 2  # type: ignore
            else (
                "moderate"
                if psi >= psi_threshold or ks_p < ks_threshold  # type: ignore
                else "none"
            )
        )

        summary = {
            "message": (
                f"Target drift: {drift_severity.upper()}"
                f" (PSI={psi:.3f}, KS p={ks_p:.3f})"
            )
        }
        recommendations = []
        if drift_severity in ("moderate", "significant"):
            recommendations.append(
                "Consider retraining or validating your model due to target drift."
            )

        return TaskResult(
            name=self.name,
            status="success",
            summary=summary,
            data={
                "target_type": "numerical",
                "psi": float(psi),
                "ks_pvalue": round(float(ks_p), 4),  # type: ignore
                "drift_rating": drift_severity,
            },
            recommendations=recommendations,
        )

    def _evaluate_categorical_drift(
        self, current: pl.Series, reference: pl.Series
    ) -> TaskResult:
        chi2_threshold = float(self.get_task_param("chi2_pvalue") or 0.05)
        entropy_threshold = float(self.get_task_param("entropy_delta") or 0.5)

        current_counts = current.value_counts().sort(current.name)
        reference_counts = reference.value_counts().sort(reference.name)

        categories = sorted(
            set(current_counts[current.name].to_list())
            | set(reference_counts[reference.name].to_list())
        )
        current_freq = {k: 0 for k in categories}
        reference_freq = {k: 0 for k in categories}

        for row in current_counts.iter_rows():
            current_freq[row[0]] = row[1]
        for row in reference_counts.iter_rows():
            reference_freq[row[0]] = row[1]

        observed = np.array([current_freq[c] for c in categories])
        expected = np.array([reference_freq[c] for c in categories])

        total_obs = observed.sum()
        total_exp = expected.sum()

        if total_obs == 0 or total_exp == 0:
            raise ValueError("Target column has no data in one of the datasets.")

        observed_pct = observed / total_obs
        expected_pct = expected / total_exp

        tvd = 0.5 * np.sum(np.abs(observed_pct - expected_pct))
        chi2_stat, chi2_p = chisquare(f_obs=observed, f_exp=expected)

        entropy_current = float(entropy(observed_pct + 1e-6))
        entropy_reference = float(entropy(expected_pct + 1e-6))
        entropy_delta = abs(entropy_current - entropy_reference)

        drift_severity = (
            "significant"
            if chi2_p < chi2_threshold / 2 or entropy_delta > 2 * entropy_threshold
            else (
                "moderate"
                if chi2_p < chi2_threshold or entropy_delta > entropy_threshold
                else "none"
            )
        )

        summary = {
            "message": (
                f"Target drift: {drift_severity.upper()}"
                f" (TVD={tvd:.3f}, Chi² p={chi2_p:.3f})"
            )
        }
        recommendations = []
        if drift_severity in ("moderate", "significant"):
            recommendations.append(
                "Class distribution drift detected."
                "Retrain or monitor model performance."
            )

        return TaskResult(
            name=self.name,
            status="success",
            summary=summary,
            data={
                "target_type": "categorical",
                "tvd": float(tvd),
                "chi2_pvalue": round(float(chi2_p), 4),
                "entropy_delta": float(entropy_delta),
                "drift_rating": drift_severity,
            },
            recommendations=recommendations,
        )
