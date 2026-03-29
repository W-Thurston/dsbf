# dsbf/eda/tasks/detect_stationarity.py
#
# ADF and KPSS stationarity tests.
#
# Epistemics - these two tests have OPPOSITE null hypotheses:
#   ADF  H₀: unit root present (non-stationary) -reject → evidence FOR stationarity
#   KPSS H₀: stationary                         -reject → evidence AGAINST stationarity
#
# This means:
#   ADF reject + KPSS not reject → consistent with stationary
#   ADF not reject + KPSS reject → consistent with non-stationary
#   Both reject                  → trend-stationary or structural break (ambiguous)
#   Neither reject               → inconclusive (insufficient data or borderline)
#
# The task never produces a single "stationary/non-stationary" verdict without
# surfacing the test agreement/disagreement. The guidance is explicit about
# what each outcome means and what to do next.

import warnings
from typing import TYPE_CHECKING, Any

from statsmodels.tsa.stattools import adfuller, kpss

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.eda.tasks._time_series_utils import (
    TimeSeriesConfig,
    TimeSeriesSetupError,
    infer_frequency,
    make_disabled_result,
    prepare_time_series_df,
    read_time_series_config,
    to_pandas,
    validate_time_series_config,
)

if TYPE_CHECKING:
    from pandas import DataFrame, Series


def _interpret_tests(
    adf_rejected: bool,
    kpss_rejected: bool,
) -> tuple[str, str]:
    """
    Return (assessment, confidence) from ADF and KPSS outcomes.

    Args:
        adf_rejected: True if ADF null (unit root) was rejected.
        kpss_rejected: True if KPSS null (stationary) was rejected.

    Returns:
        Tuple of (assessment string, confidence string).

    """
    if adf_rejected and not kpss_rejected:
        return "consistent_with_stationary", "moderate"
    if not adf_rejected and kpss_rejected:
        return "consistent_with_non_stationary", "moderate"
    if adf_rejected and kpss_rejected:
        return "ambiguous_trend_stationary_or_structural_break", "low"
    # Neither rejected
    return "inconclusive", "low"


@register_task(
    name="detect_stationarity",
    display_name="Stationarity Tests (ADF / KPSS)",
    description=(
        "Tests for stationarity using ADF (unit root test) and KPSS. "
        "The two tests have opposite null hypotheses - their agreement and "
        "disagreement are both informative. Never produces a single verdict "
        "without surfacing the full test evidence."
    ),
    depends_on=["infer_types", "temporal_gap_detection"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["time_series", "stationarity", "adf", "kpss", "unit_root"],
    expected_semantic_types=["continuous"],
)
class DetectStationarity(BaseTask):
    """
    Test for stationarity using ADF and KPSS.

    **Why two tests?**
    ADF and KPSS have opposite null hypotheses. Rejecting ADF is evidence
    *for* stationarity; rejecting KPSS is evidence *against* it. Using both
    tests together gives four informative outcomes:

    - **Both consistent**: ADF rejected, KPSS not rejected → strong evidence
      of stationarity.
    - **Both inconsistent**: ADF not rejected, KPSS rejected → strong evidence
      of non-stationarity. Differencing or detrending needed.
    - **Both reject**: Trend-stationary series or structural break. The series
      may become stationary after removing a deterministic trend.
    - **Neither rejects**: Inconclusive. Sample may be too small, or the
      series is borderline.

    **What stationarity matters for:**
    ACF/PACF and most ARIMA models assume stationarity. A non-stationary
    series produces spurious autocorrelations and unreliable model fits.
    Standard fix: first-difference (I(1)) or seasonal-difference the series,
    then re-test.

    **Epistemic note:**
    Statistical tests for stationarity have limited power on short series
    (< 50 observations). Results on short series should be treated as
    suggestive, not conclusive.

    Configurable parameters
    (via config["tasks"]["time_series"]["tasks"]["stationarity"]):
        alpha (float): Significance threshold. Default: 0.05
        tests (list[str]): Which tests to run. Default: ["adf", "kpss"]
    """

    def run(self) -> None:
        """
        Execute stationarity tests and populate self.output.

        Raises:
            Exception: Re-raised if a context is present.

        """
        try:
            df: DataFrame = to_pandas(self.input_data)

            ts_config: TimeSeriesConfig = read_time_series_config(self)

            if not ts_config.enabled:
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    **make_disabled_result(
                        self.name,
                        "Time series analysis is not enabled. Set "
                        "time_series.enabled: true and "
                        "time_series.datetime_index_column in config.",
                    ),
                )
                return

            try:
                validate_time_series_config(ts_config, df, self)
                df, value_cols = prepare_time_series_df(df, ts_config, self)
            except TimeSeriesSetupError as e:
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    **make_disabled_result(self.name, str(e)),
                )
                return

            ts_tasks_cfg: dict = self.get_shared_param("time_series", "tasks") or {}
            stat_cfg: dict = (
                ts_tasks_cfg.get("stationarity", {}) if ts_tasks_cfg else {}
            )

            alpha = float(stat_cfg.get("alpha", 0.05))
            tests_to_run: list = list(stat_cfg.get("tests", ["adf", "kpss"]))

            frequency: str | None = infer_frequency(
                df[ts_config.index_col],
                ts_config.frequency,
                self,
            )

            results: dict[str, dict[str, Any]] = {}

            for col in value_cols:
                series: Series = df[col].dropna()
                n: int = len(series)

                if n < 10:
                    self._log(
                        f"    '{col}' skipped: only {n} observations.",
                        "debug",
                    )
                    continue

                col_result: dict[str, Any] = {"n": n, "alpha": alpha}
                adf_rejected = None
                kpss_rejected = None

                # ADF test
                if "adf" in tests_to_run:
                    try:
                        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(
                            series.values,
                            autolag="AIC",
                        )
                        adf_rejected = bool(adf_p < alpha)
                        col_result["adf"] = {
                            "test_statistic": round(float(adf_stat), 6),
                            "p_value": round(float(adf_p), 6),
                            "lags_used": int(adf_lags),
                            "n_obs": int(adf_nobs),
                            "critical_values": {
                                k: round(float(v), 6) for k, v in adf_crit.items()
                            },
                            "rejected_h0": adf_rejected,
                            "interpretation": (
                                "Unit root rejected - evidence for stationarity"
                                if adf_rejected
                                else (
                                    "Unit root not rejected - "
                                    "evidence for non-stationarity"
                                )
                            ),
                        }
                    except Exception as e:
                        self._log(f"    '{col}' ADF failed: {e}", "warn")
                        col_result["adf"] = {"error": str(e)}

                # KPSS test
                if "kpss" in tests_to_run:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(
                                series.values,
                                regression="c",
                                nlags="auto",
                            )
                        kpss_rejected = bool(kpss_p < alpha)
                        col_result["kpss"] = {
                            "test_statistic": round(float(kpss_stat), 6),
                            "p_value": round(float(kpss_p), 6),
                            "lags_used": int(kpss_lags),
                            "critical_values": {
                                k: round(float(v), 6) for k, v in kpss_crit.items()
                            },
                            "rejected_h0": kpss_rejected,
                            "interpretation": (
                                "Stationarity rejected - evidence for non-stationarity"
                                if kpss_rejected
                                else (
                                    "Stationarity not rejected - "
                                    "evidence for stationarity"
                                )
                            ),
                        }
                    except Exception as e:
                        self._log(f"    '{col}' KPSS failed: {e}", "warn")
                        col_result["kpss"] = {"error": str(e)}

                # Joint assessment only when both tests ran successfully
                if adf_rejected is not None and kpss_rejected is not None:
                    assessment, confidence = _interpret_tests(
                        adf_rejected,
                        kpss_rejected,
                    )
                    col_result["assessment"] = assessment
                    col_result["confidence"] = confidence
                    col_result["caveats"] = _build_caveats(n, assessment)
                elif adf_rejected is not None:
                    col_result["assessment"] = (
                        "consistent_with_stationary"
                        if adf_rejected
                        else "consistent_with_non_stationary"
                    )
                    col_result["confidence"] = "low"
                    col_result["caveats"] = [
                        "Only ADF was run. KPSS provides complementary evidence "
                        "with an opposite null hypothesis - add 'kpss' to "
                        "time_series.tasks.stationarity.tests for a fuller picture.",
                        *_build_caveats(n, col_result["assessment"]),
                    ]
                elif kpss_rejected is not None:
                    col_result["assessment"] = (
                        "consistent_with_non_stationary"
                        if kpss_rejected
                        else "consistent_with_stationary"
                    )
                    col_result["confidence"] = "low"
                    col_result["caveats"] = [
                        "Only KPSS was run. ADF provides complementary evidence "
                        "- add 'adf' to time_series.tasks.stationarity.tests.",
                        *_build_caveats(n, col_result["assessment"]),
                    ]
                else:
                    col_result["assessment"] = "indeterminate"
                    col_result["confidence"] = "low"
                    col_result["caveats"] = ["No tests completed successfully."]

                results[col] = col_result
                self._log(
                    f"    '{col}': assessment={col_result['assessment']}, "
                    f"confidence={col_result['confidence']}",
                    "debug",
                )

            non_stationary: int = sum(
                1
                for v in results.values()
                if "non_stationary" in v.get("assessment", "")
            )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Stationarity tests run on {len(results)} column(s); "
                        f"{non_stationary} consistent with non-stationarity."
                    ),
                    "columns_tested": len(results),
                    "non_stationary_count": non_stationary,
                    "alpha": alpha,
                    "tests_run": tests_to_run,
                    "epistemic_note": (
                        "ADF and KPSS have opposite null hypotheses. Agreement "
                        "between them strengthens the conclusion; disagreement "
                        "signals ambiguity that requires domain judgment."
                    ),
                },
                data=results,
                metadata={
                    "index_column": ts_config.index_col,
                    "frequency": frequency,
                    "alpha": alpha,
                    "tests_run": tests_to_run,
                    "suggested_viz_type": "table",
                    "recommended_section": "Time Series",
                    "display_priority": "high",
                    "column_types": self.get_column_type_info(value_cols),
                },
            )

            for col, col_data in results.items():
                self._attach_guidance(col, col_data)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, col_data: dict[str, Any]) -> None:
        """
        Generate EDA and ML guidance for stationarity findings.

        Args:
            col: Column name.
            col_data: Stationarity result dict for this column.

        """
        assessment = col_data.get("assessment", "indeterminate")
        confidence = col_data.get("confidence", "low")
        caveats = col_data.get("caveats", [])
        n = col_data["n"]
        alpha = col_data["alpha"]

        adf = col_data.get("adf", {})
        kpss_res = col_data.get("kpss", {})

        adf_str: str = (
            f"ADF p={adf['p_value']:.4f} "
            f"({'rejected' if adf.get('rejected_h0') else 'not rejected'})"
            if "p_value" in adf
            else "ADF: not run"
        )
        kpss_str: str = (
            f"KPSS p={kpss_res['p_value']:.4f} "
            f"({'rejected' if kpss_res.get('rejected_h0') else 'not rejected'})"
            if "p_value" in kpss_res
            else "KPSS: not run"
        )

        assessment_display = assessment.replace("_", " ").title()
        caveats_str: str = "\n".join(f"⚠ {c}" for c in caveats)

        eda_body: str = (
            f"'{col}' stationarity assessment: {assessment_display} "
            f"(confidence: {confidence}, n={n:,}, α={alpha}).\n\n"
            f"{adf_str}; {kpss_str}.\n\n"
            + (f"Caveats:\n{caveats_str}" if caveats else "")
        )

        level: str = (
            "warn"
            if "non_stationary" in assessment
            or assessment == "ambiguous_trend_stationary_or_structural_break"
            else "info"
        )

        actions: list = []
        if "non_stationary" in assessment:
            actions = [
                {
                    "action": "difference_series",
                    "detail": (
                        "Apply first differencing: df[col].diff().dropna() "
                        "then re-run stationarity tests"
                    ),
                },
                {
                    "action": "log_transform_then_difference",
                    "detail": (
                        "For exponential trends: log-transform first, then difference"
                    ),
                    "condition": "if series has exponential growth pattern",
                },
            ]
        elif "ambiguous" in assessment:
            actions = [
                {
                    "action": "remove_deterministic_trend",
                    "detail": (
                        "Fit and subtract a linear trend (OLS), "
                        "then re-run stationarity tests on residuals"
                    ),
                },
            ]

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=level,
            title=f"Stationarity: {assessment_display} (confidence: {confidence})",
            body=eda_body.strip(),
            actions=actions,
            metric={
                "assessment": assessment,
                "confidence": confidence,
                "n": n,
                "adf_p_value": adf.get("p_value"),
                "kpss_p_value": kpss_res.get("p_value"),
            },
        )


def _build_caveats(n: int, assessment: str) -> list[str]:
    """
    Build standard caveats list for a stationarity assessment.

    Args:
        n: Number of observations.
        assessment: Assessment string.

    Returns:
        List of caveat strings.

    """
    caveats: list[str] = []
    if n < 50:
        caveats.append(
            f"Series is short (n={n}). Both ADF and KPSS have low power "
            "on small samples - results should be treated as suggestive.",
        )
    if "ambiguous" in assessment:
        caveats.append(
            "Both tests rejected their null hypotheses. This can indicate a "
            "trend-stationary series (stationary around a deterministic trend) "
            "or a structural break. Try removing a linear trend and re-testing.",
        )
    if "inconclusive" in assessment:
        caveats.append(
            "Neither test rejected. The series may be borderline, or the "
            "sample may be too small to distinguish stationary from "
            "non-stationary behaviour.",
        )
    caveats.append(
        "These tests assume no structural breaks. A series can appear "
        "non-stationary due to a single break point rather than a genuine "
        "unit root - inspect the time plot before differencing.",
    )
    return caveats
