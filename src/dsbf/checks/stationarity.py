# dsbf/checks/stationarity.py
"""Numeric columns whose level drifts over time, by ADF and KPSS on a regular series."""

import math
from typing import Any, Literal

import numpy as np
from pydantic import Field
from statsmodels.tsa.stattools import adfuller, kpss

from dsbf.checks.timeseries import (
    Prepared,
    SeriesInfo,
    TimeSeriesOutput,
    TimeSeriesParams,
    prepare,
)
from dsbf.engine import (
    CheckContext,
    CheckOutput,
    Requires,
    SemanticTypes,
    check,
)
from dsbf.model import (
    Confidence,
    Contract,
    Cost,
    Dimension,
    Severity,
    Subject,
)

Assessment = Literal[
    "consistent_with_stationary",
    "consistent_with_trend_stationary",
    "consistent_with_non_stationary",
    "ambiguous_trend_stationary_or_structural_break",
    "inconclusive",
    "indeterminate",
]

# A failed test: statsmodels raises these for a series it cannot fit.
_FAILURES = (ValueError, RuntimeError, np.linalg.LinAlgError)
# KPSS interpolates its p-value in a table that runs from 0.01 to 0.1.
_KPSS_TABLE = (0.01, 0.1)

_MEANING: dict[Assessment, str] = {
    "consistent_with_stationary": "both tests point to a stationary series",
    "consistent_with_trend_stationary": (
        "with a constant alone the tests say the level drifts, but once a straight "
        "trend is allowed for, ADF rejects a unit root and KPSS does not reject "
        "stationarity: the series is stationary around a deterministic trend"
    ),
    "consistent_with_non_stationary": (
        "ADF cannot reject a unit root and KPSS rejects stationarity: the level "
        "drifts, as with a trend or a random walk"
    ),
    "ambiguous_trend_stationary_or_structural_break": (
        "both tests reject their null, as for a series that is stationary around a "
        "trend or has a structural break"
    ),
    "inconclusive": "neither test rejects its null, so the data cannot say",
    "indeterminate": "a test could not be run",
}


class StationarityParams(TimeSeriesParams):
    alpha: float = Field(
        0.05, gt=0, lt=1, description="Significance level of both tests."
    )
    max_lag: int = Field(
        24,
        ge=1,
        le=365,
        description="Most lags of the ADF regression, from which AIC picks, and of "
        "the KPSS long-run variance. statsmodels' own default grows with the series "
        "and every candidate lag is a regression.",
    )
    min_periods: int = Field(
        20, ge=10, description="Periods a series needs to be tested."
    )


class UnitRootTest(Contract):
    test: Literal["adf", "kpss"]
    statistic: float
    p_value: float = Field(ge=0, le=1)
    p_value_limit: Literal["at_most", "at_least"] | None = Field(
        description="KPSS reads its p-value from a table between 0.01 and 0.1; "
        "outside it the value is a bound, not an estimate."
    )
    lags: int = Field(ge=0, description="Lags the test used.")
    critical_values: dict[str, float]
    rejects: bool = Field(description="p is below `alpha`.")


class ColumnStationarity(SeriesInfo):
    adf: UnitRootTest | None = Field(
        description="Null hypothesis: a unit root. Rejecting it favours stationarity."
    )
    kpss: UnitRootTest | None = Field(
        description="Null hypothesis: stationary around a level. Rejecting it "
        "favours a drifting level."
    )
    adf_trend: UnitRootTest | None = Field(
        description="ADF with a constant and a linear trend in the regression."
    )
    kpss_trend: UnitRootTest | None = Field(
        description="Null hypothesis: stationary around a linear trend."
    )
    errors: dict[str, str] = Field(description="Tests that could not run, and why.")
    assessment: Assessment
    confidence: Literal["moderate", "low"]
    max_lag: int = Field(ge=0, description="The cap on lags for this series' length.")


class StationarityOutput(TimeSeriesOutput):
    alpha: float
    columns: dict[str, ColumnStationarity] = Field(
        description="Every column that was tested, by name."
    )


@check(
    id="stationarity",
    version="1.1.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric", "datetime"),
    dimensions=[Dimension.TEMPORAL],
    params=StationarityParams,
    output=StationarityOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.MODERATE,
    extras=["timeseries"],
    packs=["core"],
    tags=["time-series", "stationarity", "adf", "kpss"],
)
def stationarity(ctx: CheckContext, params: StationarityParams) -> CheckOutput:
    """Test each numeric column for a unit root and a drifting level over time.

    The columns are resampled to the frequency first (see ``checks.timeseries``), the
    most recent ``max_periods`` periods are used, and the lags are capped.
    """
    prepared = prepare(ctx, params, min_periods=params.min_periods)
    if isinstance(prepared, CheckOutput):
        return prepared
    columns: dict[str, ColumnStationarity] = {}
    for name, series in prepared.series.items():
        ctx.checkpoint()
        result = _test(series.values, params, prepared.info(name))
        columns[name] = result
        if result.assessment == "consistent_with_non_stationary":
            _report(ctx, name, result, params, prepared, "non_stationary")
        elif result.assessment == "consistent_with_trend_stationary":
            _report(ctx, name, result, params, prepared, "trend_stationary")
        elif result.assessment == "ambiguous_trend_stationary_or_structural_break":
            _report(ctx, name, result, params, prepared, "ambiguous_stationarity")
    return ctx.output(
        prepared.build(StationarityOutput, alpha=params.alpha, columns=columns)
    )


def _test(
    values: np.ndarray, params: StationarityParams, info: SeriesInfo
) -> ColumnStationarity:
    n = int(values.size)
    # The ADF regression needs `maxlag <= n // 2 - 2` with a constant.
    cap = max(0, min(params.max_lag, n // 2 - 2))
    errors: dict[str, str] = {}
    adf = _adf(values, cap, params.alpha, errors)
    kpss_test = _kpss(values, cap, params.alpha, errors)
    adf_trend = _adf(values, cap, params.alpha, errors, regression="ct")
    kpss_trend = _kpss(values, cap, params.alpha, errors, regression="ct")
    assessment, confidence = _assess(adf, kpss_test)
    if (
        assessment
        in (
            "consistent_with_non_stationary",
            "ambiguous_trend_stationary_or_structural_break",
        )
        and adf_trend is not None
        and kpss_trend is not None
        and adf_trend.rejects
        and not kpss_trend.rejects
    ):
        assessment, confidence = "consistent_with_trend_stationary", "moderate"
    return ColumnStationarity(
        **info.model_dump(),
        adf=adf,
        kpss=kpss_test,
        adf_trend=adf_trend,
        kpss_trend=kpss_trend,
        errors=errors,
        assessment=assessment,
        confidence=confidence,
        max_lag=cap,
    )


def _adf(
    values: np.ndarray,
    cap: int,
    alpha: float,
    errors: dict[str, str],
    regression: Literal["c", "ct"] = "c",
) -> UnitRootTest | None:
    try:
        found: Any = adfuller(values, maxlag=cap, regression=regression, autolag="AIC")
    except _FAILURES as error:
        errors["adf" if regression == "c" else "adf_trend"] = str(error)
        return None
    statistic, p_value, lags, critical = found[0], found[1], found[2], found[4]
    return _result("adf", statistic, p_value, lags, critical, alpha, errors)


def _kpss(
    values: np.ndarray,
    cap: int,
    alpha: float,
    errors: dict[str, str],
    regression: Literal["c", "ct"] = "c",
) -> UnitRootTest | None:
    n = values.size
    # The long-run variance of the Newey–West estimate; statsmodels' automatic choice
    # is capped at the same number of lags as the ADF regression.
    lags = min(cap, math.ceil(12 * (n / 100) ** 0.25))
    try:
        found: Any = kpss(values, regression=regression, nlags=lags)
    except _FAILURES as error:
        errors["kpss" if regression == "c" else "kpss_trend"] = str(error)
        return None
    statistic, p_value, used, critical = found[0], found[1], found[2], found[3]
    return _result("kpss", statistic, p_value, used, critical, alpha, errors)


def _result(
    test: Literal["adf", "kpss"],
    statistic: float,
    p_value: float,
    lags: int,
    critical: dict[str, float],
    alpha: float,
    errors: dict[str, str],
) -> UnitRootTest | None:
    numbers = [statistic, p_value, *critical.values()]
    if not all(np.isfinite(numbers)):
        errors[test] = "the test statistic is not finite"
        return None
    limit = None
    if test == "kpss":
        if p_value <= _KPSS_TABLE[0]:
            limit = "at_most"
        elif p_value >= _KPSS_TABLE[1]:
            limit = "at_least"
    return UnitRootTest(
        test=test,
        statistic=float(statistic),
        p_value=float(p_value),
        p_value_limit=limit,
        lags=int(lags),
        critical_values={key: float(value) for key, value in critical.items()},
        rejects=bool(p_value < alpha),
    )


def _assess(
    adf: UnitRootTest | None, kpss_test: UnitRootTest | None
) -> tuple[Assessment, Literal["moderate", "low"]]:
    """The joint reading of the two tests, which have opposite null hypotheses."""
    if adf is None or kpss_test is None:
        return "indeterminate", "low"
    if adf.rejects and not kpss_test.rejects:
        return "consistent_with_stationary", "moderate"
    if not adf.rejects and kpss_test.rejects:
        return "consistent_with_non_stationary", "moderate"
    if adf.rejects and kpss_test.rejects:
        return "ambiguous_trend_stationary_or_structural_break", "low"
    return "inconclusive", "low"


def _report(
    ctx: CheckContext,
    name: str,
    result: ColumnStationarity,
    params: StationarityParams,
    prepared: Prepared,
    code: Literal["non_stationary", "trend_stationary", "ambiguous_stationarity"],
) -> None:
    assert result.adf is not None and result.kpss is not None
    metrics: dict[str, object] = {
        "alpha": params.alpha,
        "assessment": result.assessment,
        "adf_statistic": result.adf.statistic,
        "adf_p_value": result.adf.p_value,
        "adf_lags": result.adf.lags,
        "kpss_statistic": result.kpss.statistic,
        "kpss_p_value": result.kpss.p_value,
        "kpss_lags": result.kpss.lags,
        "maxlag": result.max_lag,
    }
    drifting = code == "non_stationary"
    around_trend = code == "trend_stationary"
    if around_trend and result.adf_trend and result.kpss_trend:
        metrics |= {
            "adf_trend_p_value": result.adf_trend.p_value,
            "kpss_trend_p_value": result.kpss_trend.p_value,
        }
    title = (
        "Non-stationary over time"
        if drifting
        else "Stationary around a trend"
        if around_trend
        else "Stationarity tests disagree"
    )
    advice = (
        "Autocorrelations and ARIMA-type models assume a stationary series, "
        "so difference or detrend it and test again."
        if drifting
        else "Remove the trend (regress on time) rather than differencing, which "
        "adds noise to a series that has no unit root; a model with a time term "
        "handles it, and a random train-test split does not."
        if around_trend
        else "Test the series after removing a deterministic trend, and "
        "look for a break in level."
    )
    ctx.finding(
        code=code,
        subject=Subject.column(name),
        severity=Severity.WARN if drifting else Severity.INFO,
        confidence=Confidence.MODERATE if drifting or around_trend else Confidence.LOW,
        title=title,
        message=(
            f"{name!r} against {prepared.index!r}, as {result.periods:,} "
            f"{prepared.step.alias} periods: {_MEANING[result.assessment]} "
            f"(ADF p={result.adf.p_value:.3g}, KPSS p={result.kpss.p_value:.3g}, "
            f"α={params.alpha:g}). "
            + advice
            + (
                f" Only the last {prepared.max_periods:,} of "
                f"{prepared.periods_total:,} periods were used."
                if prepared.truncated
                else ""
            )
        ),
        evidence=prepared.evidence(ctx, name, metrics, "adf+kpss"),
    )
