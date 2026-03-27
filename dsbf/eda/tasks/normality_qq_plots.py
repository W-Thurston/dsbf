# dsbf/eda/tasks/normality_qq_plots.py

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

if TYPE_CHECKING:
    from numpy import ndarray

# ── QQ data computation ────────────────────────────────────────────────────────


def _compute_qq_data(
    series: pd.Series,
    n_quantiles: int = 100,
) -> dict[str, Any]:
    """
    Compute theoretical vs empirical quantile pairs for a normal QQ plot.

    Fits a normal distribution to the data (using mean and std), then
    computes paired quantile values at evenly spaced probability points.
    Also computes the reference line (the line a perfectly normal
    distribution would follow) as two anchor points.

    Args:
        series: Non-null numeric Series.
        n_quantiles: Number of quantile points to compute. Default: 100.

    Returns:
        Dict with ``theoretical``, ``empirical``, ``reference_line``,
        ``fit_mean``, ``fit_std``, and ``n`` keys.

    """
    values: ndarray = np.sort(series.dropna().to_numpy())
    n: int = len(values)

    if n < 4:
        return {}

    # Probability points: avoid 0 and 1 to prevent infinite quantiles
    probabilities: ndarray = np.linspace(1 / (n + 1), n / (n + 1), min(n_quantiles, n))

    # Fit normal parameters from data
    fit_mean = float(np.mean(values))
    fit_std = float(np.std(values, ddof=1))

    if fit_std == 0:
        return {}

    # Theoretical quantiles from fitted normal
    theoretical: ndarray = stats.norm.ppf(probabilities, loc=fit_mean, scale=fit_std)

    # Empirical quantiles from actual data
    empirical = np.quantile(values, probabilities)

    # Reference line: passes through Q1 and Q3 of the fitted normal
    # (standard QQ plot convention)
    q1_theoretical: ndarray = stats.norm.ppf(0.25, loc=fit_mean, scale=fit_std)
    q3_theoretical: ndarray = stats.norm.ppf(0.75, loc=fit_mean, scale=fit_std)
    q1_empirical = float(np.quantile(values, 0.25))
    q3_empirical = float(np.quantile(values, 0.75))

    return {
        "theoretical": [round(float(v), 6) for v in theoretical],
        "empirical": [round(float(v), 6) for v in empirical],
        "reference_line": {
            "x": [round(float(q1_theoretical), 6), round(float(q3_theoretical), 6)],
            "y": [round(float(q1_empirical), 6), round(float(q3_empirical), 6)],
        },
        "fit_mean": round(fit_mean, 6),
        "fit_std": round(fit_std, 6),
        "n": n,
        "n_quantiles": len(probabilities),
    }


def _deviation_summary(theoretical: list, empirical: list) -> dict[str, Any]:
    """
    Summarise how far the empirical quantiles deviate from the reference line.

    Computes the mean absolute deviation between empirical and theoretical
    quantiles, and identifies the tail regions (bottom/top 10%) where
    deviations are most informative.

    Args:
        theoretical: List of theoretical quantile values.
        empirical: List of empirical quantile values.

    Returns:
        Dict with ``mean_abs_deviation``, ``tail_deviation_lower``,
        ``tail_deviation_upper``, and ``max_deviation`` keys.

    """
    if not theoretical or not empirical:
        return {}

    t: ndarray = np.array(theoretical)
    e: ndarray = np.array(empirical)
    deviations = np.abs(e - t)
    n: int = len(deviations)
    tail_n: int = max(1, n // 10)

    return {
        "mean_abs_deviation": round(float(np.mean(deviations)), 6),
        "max_deviation": round(float(np.max(deviations)), 6),
        "tail_deviation_lower": round(float(np.mean(deviations[:tail_n])), 6),
        "tail_deviation_upper": round(float(np.mean(deviations[-tail_n:])), 6),
    }


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="normality_qq_plots",
    display_name="Normality QQ Plots",
    description=(
        "Computes theoretical vs empirical quantile pairs for normal QQ plots. "
        "Prioritises columns that rejected normality in normality_tests. "
        "Output feeds the frontend QQ plot renderer."
    ),
    depends_on=["infer_types", "normality_tests"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["distribution", "normality", "visualization", "numeric"],
    expected_semantic_types=["continuous"],
)
class NormalityQQPlots(BaseTask):
    """
    Compute QQ plot data for continuous columns to visualise normality deviations.

    A QQ (quantile-quantile) plot compares the empirical distribution of a
    column against a theoretical normal distribution. Points that fall on the
    diagonal reference line indicate normality; systematic deviations reveal
    the specific pattern of non-normality:

    - **S-shaped curve**: heavy tails (leptokurtic)
    - **Inverse S-curve**: thin tails (platykurtic)
    - **Concave/convex curve**: skewness
    - **Step pattern**: discrete or bounded data

    The normality test (Shapiro-Wilk, KS, Jarque-Bera) gives a binary
    verdict; the QQ plot explains *where* the distribution deviates and *how*.

    **Column selection:**
    By default, only columns that rejected normality in ``normality_tests``
    are processed. Set ``include_normal=True`` to include all continuous
    columns. Columns not covered by ``normality_tests`` (e.g. if that task
    didn't run) fall back to the ``skew_threshold`` heuristic.

    **Output format:**
    Each column result contains ``theoretical`` and ``empirical`` lists of
    equal length - paired quantile values ready for direct plotting - plus
    a ``reference_line`` with two anchor points (Q1 and Q3) and a
    ``deviation_summary`` with quantified departure statistics.

    The output is consumed by the frontend QQ plot renderer in the
    Distributions tab. No static image is generated by this task.

    Configurable parameters (via config["tasks"]["normality_qq_plots"]):
        include_normal (bool): Also compute QQ data for columns that passed
            normality tests. Default: False
        n_quantiles (int): Number of quantile points per plot. Default: 100
        skew_threshold (float): Fallback - compute QQ for columns with
            |skewness| ≥ this value when normality_tests has not run.
            Default: 0.5
        min_n (int): Minimum non-null values to compute QQ data. Default: 8
    """

    def run(self) -> None:
        """
        Execute QQ data computation and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'continuous' column(s)",
                "debug",
            )

            include_normal_raw: Any | None = self.get_task_param("include_normal")
            include_normal: bool = (
                str(include_normal_raw).lower() in ("true", "1", "yes")
                if include_normal_raw is not None
                else False
            )

            n_quantiles_raw: Any | None = self.get_task_param("n_quantiles")
            n_quantiles: int = (
                int(n_quantiles_raw) if n_quantiles_raw is not None else 100
            )

            skew_thresh_raw: Any | None = self.get_task_param("skew_threshold")
            skew_threshold: float = (
                float(skew_thresh_raw) if skew_thresh_raw is not None else 0.5
            )

            min_n_raw: Any | None = self.get_task_param("min_n")
            min_n: int = int(min_n_raw) if min_n_raw is not None else 8

            # --- Determine which columns to process ---
            target_cols: set[str] = set()
            normality_verdicts: dict[str, str] = {}
            normality_source = "none"

            if self.context:
                norm_result: TaskResult | None = self.context.results.get(
                    "normality_tests",
                )
                if norm_result and norm_result.status == "success":
                    normality_source = "normality_tests"
                    for col, info in (norm_result.data or {}).items():
                        verdict = info.get("overall_verdict", "unknown")
                        normality_verdicts[col] = verdict
                        if verdict == "non_normal" or include_normal:
                            target_cols.add(col)

            if not target_cols:
                # Fallback: use skewness threshold on numeric columns
                self._log(
                    "    No normality_tests result in context - using "
                    "skewness fallback.",
                    "debug",
                )
                normality_source = "skewness_fallback"
                numeric_df = df.select_dtypes(include=np.number)
                for col in numeric_df.columns:
                    series = numeric_df[col].dropna()
                    if len(series) >= min_n:
                        try:
                            sk = float(series.skew())
                            if abs(sk) >= skew_threshold or include_normal:
                                target_cols.add(col)
                        except Exception:
                            pass

            self._log(
                f"    {len(target_cols)} column(s) selected for QQ computation "
                f"(source: {normality_source}, include_normal={include_normal}).",
                "debug",
            )

            qq_data: dict[str, dict[str, Any]] = {}

            for col in target_cols:
                if col not in df.columns:
                    continue

                series = df[col].dropna()
                if len(series) < min_n:
                    self._log(
                        f"    '{col}' skipped: only {len(series)} non-null values.",
                        "debug",
                    )
                    continue

                if not pd.api.types.is_numeric_dtype(series):
                    continue

                qq: dict[str, Any] = _compute_qq_data(series, n_quantiles=n_quantiles)
                if not qq:
                    continue

                deviation: dict[str, Any] = _deviation_summary(
                    qq["theoretical"],
                    qq["empirical"],
                )

                qq_data[col] = {
                    **qq,
                    "deviation_summary": deviation,
                    "normality_verdict": normality_verdicts.get(col, "unknown"),
                }

                self._log(
                    f"    '{col}': {qq['n_quantiles']} quantile points computed, "
                    f"mean_abs_deviation="
                    f"{deviation.get('mean_abs_deviation', 'n/a'):.4f}",
                    "debug",
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (f"QQ plot data computed for {len(qq_data)} column(s)."),
                    "columns_computed": len(qq_data),
                    "normality_source": normality_source,
                    "include_normal": include_normal,
                },
                data=qq_data,
                metadata={
                    "n_quantiles": n_quantiles,
                    "skew_threshold": skew_threshold,
                    "include_normal": include_normal,
                    "suggested_viz_type": "scatter",
                    "recommended_section": "Distributions",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, data in qq_data.items():
                self._attach_guidance(col, data)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, data: dict[str, Any]) -> None:
        """
        Generate EDA guidance interpreting the QQ plot deviation pattern.

        Args:
            col: Column name.
            data: QQ data dict including deviation_summary and normality_verdict.

        """
        deviation = data.get("deviation_summary", {})
        verdict = data.get("normality_verdict", "unknown")
        n = data.get("n", 0)
        mean_dev = deviation.get("mean_abs_deviation", 0.0)
        tail_lower = deviation.get("tail_deviation_lower", 0.0)
        tail_upper = deviation.get("tail_deviation_upper", 0.0)
        max_dev = deviation.get("max_deviation", 0.0)

        # Characterise the deviation pattern from tail behaviour
        if tail_lower > mean_dev * 1.5 and tail_upper > mean_dev * 1.5:
            pattern = "heavy tails (both ends deviate) - consistent with high kurtosis"
        elif tail_upper > mean_dev * 1.5:
            pattern = (
                "right tail deviation - consistent with right skew or upper outliers"
            )
        elif tail_lower > mean_dev * 1.5:
            pattern = (
                "left tail deviation - consistent with left skew or lower outliers"
            )
        elif max_dev < 0.1:
            pattern = "minor deviation - distribution is approximately normal"
        else:
            pattern = "moderate overall deviation"

        verdict_note: str = (
            " Normality was rejected by statistical tests."
            if verdict == "non_normal"
            else (
                " Normality was not rejected by "
                f"statistical tests (verdict: {verdict})."
            )
        )

        body: str = (
            f"QQ plot data computed for '{col}' (n={n:,}). {verdict_note} "
            f"Deviation pattern: {pattern}. "
            f"Mean absolute deviation from reference line: {mean_dev:.4f}. "
            f"Lower tail deviation: {tail_lower:.4f}, "
            f"upper tail deviation: {tail_upper:.4f}. "
            f"Points near the reference line indicate normality; "
            f"systematic curves reveal skewness (concave/convex), "
            f"heavy tails (S-shape), or light tails (inverse S-shape). "
            f"See the Distributions tab to view the interactive QQ plot."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="info" if verdict != "non_normal" else "warn",
            title=f"QQ Plot: {pattern.split(' -')[0].title()}",
            body=body.strip(),
            actions=[],
            metric={
                "mean_abs_deviation": mean_dev,
                "tail_deviation_lower": tail_lower,
                "tail_deviation_upper": tail_upper,
                "max_deviation": max_dev,
                "normality_verdict": verdict,
                "n": n,
            },
        )
