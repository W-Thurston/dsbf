# dsbf/checks/autocorrelation.py
"""Numeric columns whose value is predictable from earlier periods: ACF and PACF."""

import numpy as np
from pydantic import Field
from statsmodels.tsa.stattools import acf, pacf

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
    Cost,
    Dimension,
    Severity,
    Subject,
)

# Flagged lags listed in a finding's evidence.
_LISTED = 10
# From here, in ACF at lag 1, the autocorrelation is a slow decay rather than a lag.
_SLOW_DECAY = 0.9


class AutocorrelationParams(TimeSeriesParams):
    alpha: float = Field(
        0.05,
        gt=0,
        lt=1,
        description="Significance level of the confidence bands: Bartlett's for the "
        "ACF, ±z/√n for the PACF.",
    )
    max_lags: int = Field(
        40,
        ge=2,
        le=365,
        description="Most lags computed; n periods allow at most n // 2 - 1.",
    )
    min_abs: float = Field(
        0.2,
        ge=0,
        lt=1,
        description="Smallest |autocorrelation| a lag is flagged at, besides lying "
        "outside the band: over many periods a negligible lag is significant.",
    )
    min_periods: int = Field(
        20, ge=10, description="Periods a series needs to be analysed."
    )


class ColumnAutocorrelation(SeriesInfo):
    max_lags: int = Field(ge=1, description="Lags computed for this series' length.")
    acf: tuple[float, ...] = Field(description="Autocorrelation at lags 0 to max_lags.")
    acf_band: tuple[float, ...] = Field(
        description="Half-width of the confidence band around zero at each lag."
    )
    pacf: tuple[float, ...] = Field(
        description="Partial autocorrelation at lags 0 to max_lags."
    )
    pacf_band: float = Field(gt=0, description="Half-width of the PACF's band.")
    significant_acf: tuple[int, ...] = Field(
        description="Lags from 1 whose ACF is outside the band."
    )
    significant_pacf: tuple[int, ...] = Field(
        description="Lags from 1 whose PACF is outside the band."
    )
    flagged_acf: tuple[int, ...] = Field(
        description="Significant ACF lags with |ACF| of at least `min_abs`."
    )
    flagged_pacf: tuple[int, ...] = Field(
        description="Significant PACF lags with |PACF| of at least `min_abs`."
    )
    strongest_lag: int | None = Field(
        description="The flagged ACF lag with the largest |ACF|; None without one."
    )


class AutocorrelationOutput(TimeSeriesOutput):
    alpha: float
    min_abs: float
    columns: dict[str, ColumnAutocorrelation] = Field(
        description="Every column that was analysed, by name."
    )


@check(
    id="autocorrelation",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric", "datetime"),
    dimensions=[Dimension.TEMPORAL],
    params=AutocorrelationParams,
    output=AutocorrelationOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.MODERATE,
    extras=["timeseries"],
    packs=["core"],
    tags=["time-series", "autocorrelation", "acf", "pacf"],
)
def autocorrelation(ctx: CheckContext, params: AutocorrelationParams) -> CheckOutput:
    """Compute the ACF and PACF of each numeric column on a regular series.

    The columns are resampled to the frequency first (see ``checks.timeseries``), the
    most recent ``max_periods`` periods are used, and the lags are capped.
    """
    prepared = prepare(ctx, params, min_periods=params.min_periods)
    if isinstance(prepared, CheckOutput):
        return prepared
    columns: dict[str, ColumnAutocorrelation] = {}
    for name, series in prepared.series.items():
        ctx.checkpoint()
        result = _analyse(series.values, params, prepared.info(name))
        columns[name] = result
        if result.flagged_acf:
            _report(ctx, name, result, params, prepared)
    return ctx.output(
        prepared.build(
            AutocorrelationOutput,
            alpha=params.alpha,
            min_abs=params.min_abs,
            columns=columns,
        )
    )


def _analyse(
    values: np.ndarray, params: AutocorrelationParams, info: SeriesInfo
) -> ColumnAutocorrelation:
    lags = max(1, min(params.max_lags, values.size // 2 - 1))
    centred = values - values.mean()
    values_acf, interval = acf(centred, nlags=lags, alpha=params.alpha, fft=True)
    values_pacf, pacf_interval = pacf(
        centred, nlags=lags, alpha=params.alpha, method="ywm"
    )
    acf_band = interval[:, 1] - values_acf
    pacf_band = float((pacf_interval[1:, 1] - values_pacf[1:]).mean())
    significant_acf = _outside(values_acf, acf_band)
    significant_pacf = _outside(values_pacf, np.full(values_pacf.size, pacf_band))
    flagged_acf = tuple(
        k for k in significant_acf if abs(values_acf[k]) >= params.min_abs
    )
    flagged_pacf = tuple(
        k for k in significant_pacf if abs(values_pacf[k]) >= params.min_abs
    )
    strongest = max(flagged_acf, key=lambda k: abs(values_acf[k]), default=None)
    return ColumnAutocorrelation(
        **info.model_dump(),
        max_lags=lags,
        acf=_round(values_acf),
        acf_band=_round(acf_band),
        pacf=_round(values_pacf),
        pacf_band=pacf_band,
        significant_acf=significant_acf,
        significant_pacf=significant_pacf,
        flagged_acf=flagged_acf,
        flagged_pacf=flagged_pacf,
        strongest_lag=strongest,
    )


def _outside(values: np.ndarray, band: np.ndarray) -> tuple[int, ...]:
    """The lags from 1 whose value is farther from zero than its band."""
    return tuple(int(k) for k in np.flatnonzero(np.abs(values) > band) if k >= 1)


def _round(values: np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(v), 6) for v in values)


def _report(
    ctx: CheckContext,
    name: str,
    result: ColumnAutocorrelation,
    params: AutocorrelationParams,
    prepared: Prepared,
) -> None:
    strongest = result.strongest_lag
    assert strongest is not None  # a flagged lag exists
    value = result.acf[strongest]
    slow = result.acf[1] >= _SLOW_DECAY
    metrics: dict[str, object] = {
        "alpha": params.alpha,
        "min_abs": params.min_abs,
        "maxlag": result.max_lags,
        "acf_lag1": result.acf[1],
        "pacf_lag1": result.pacf[1],
        "strongest_lag": strongest,
        "strongest_acf": value,
        "n_flagged_acf": len(result.flagged_acf),
        "flagged_acf": result.flagged_acf[:_LISTED],
        "n_flagged_pacf": len(result.flagged_pacf),
        "flagged_pacf": result.flagged_pacf[:_LISTED],
    }
    ctx.finding(
        code="autocorrelated",
        subject=Subject.column(name),
        severity=Severity.INFO,
        confidence=Confidence.MODERATE,
        title=f"Autocorrelated: lag {strongest} has ACF {value:+.2f}",
        message=(
            f"{name!r} against {prepared.index!r}, as {result.periods:,} "
            f"{prepared.step.alias} periods: {len(result.flagged_acf)} of "
            f"{result.max_lags} lags have an ACF outside the {1 - params.alpha:.0%} "
            f"band and of at least {params.min_abs:g}, the strongest at lag "
            f"{strongest} ({value:+.2f}). "
            + (
                "The ACF at lag 1 is "
                f"{result.acf[1]:.2f} and decays slowly, which a trend or a unit "
                "root also produces; check stationarity before reading the lags. "
                if slow
                else ""
            )
            + "Observations that are not independent break the assumption of "
            "most tests and of a random train-test split: split by time, and "
            "consider lagged features or a time-series model."
        ),
        evidence=prepared.evidence(ctx, name, metrics, "acf+pacf"),
    )
