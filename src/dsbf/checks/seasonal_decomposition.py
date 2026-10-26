# dsbf/checks/seasonal_decomposition.py
"""Numeric columns with a trend or a seasonal cycle, by STL on a regular series."""

import numpy as np
from pydantic import Field

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
    CheckStatus,
    Confidence,
    Cost,
    Dimension,
    Severity,
    Subject,
)

# Periods in one cycle of the next longer natural unit, for one step of each unit: a
# day has 24 hours, a week 7 days, a year 52 weeks, 12 months or 4 quarters. A step of
# several units divides it (6 hours: 4 a day), and one that does not divide it, or
# leaves a single period, implies no cycle. A year has none.
_CYCLE: dict[str, int] = {
    "min": 60,
    "h": 24,
    "D": 7,
    "W": 52,
    "MS": 12,
    "ME": 12,
    "QS": 4,
    "QE": 4,
}
# Strength from which a component is called strong or moderate, as the task did.
_STRONG = 0.7
_MODERATE = 0.4
# What white noise reaches. STL fits a seasonal component and a trend to any series, so
# even noise has a strength, more of it the fewer cycles and periods there are. The
# 99th percentile of 150 white-noise series of each length (periods of 4 to 52, 24 to
# 2 000 values) is within 0.05 of these, which are slightly conservative.
_SEASONAL_FLOOR = (0.20, 0.90)  # 0.20 + 0.90 / sqrt(cycles)
_TREND_FLOOR = (0.12, 1.50)  # 0.12 + 1.50 / sqrt(periods)


class SeasonalDecompositionParams(TimeSeriesParams):
    period: int | None = Field(
        None,
        ge=2,
        le=10_000,
        description="Periods in one seasonal cycle. Unset, it follows from the "
        "frequency: 7 for daily data, 12 for monthly, 4 for quarterly, 24 for hourly.",
    )
    robust: bool = Field(
        True, description="Fit STL robustly, so outliers do not bend the components."
    )
    min_cycles: int = Field(
        2, ge=2, description="Whole cycles of the period a series needs."
    )
    min_strength: float = Field(
        0.4,
        ge=0,
        le=1,
        description="Smallest trend or seasonal strength that is reported: a "
        "moderate one. A series with few cycles or periods must also beat what "
        "white noise of its length reaches by chance, which is higher.",
    )


class ColumnDecomposition(SeriesInfo):
    period: int = Field(ge=2, description="Periods in one seasonal cycle.")
    robust: bool
    trend_strength: float = Field(
        ge=0,
        le=1,
        description="1 - Var(remainder) / Var(trend + remainder), at least 0 "
        "(Wang, Smith and Hyndman 2006).",
    )
    seasonal_strength: float = Field(
        ge=0,
        le=1,
        description="1 - Var(remainder) / Var(seasonal + remainder), at least 0.",
    )
    remainder_share: float = Field(
        ge=0, description="Variance of the remainder over the variance of the series."
    )
    seasonal_floor: float = Field(
        ge=0,
        le=1,
        description="The seasonal strength that white noise of this many cycles "
        "reaches in 1% of series: a seasonal strength below it is not reported.",
    )
    trend_floor: float = Field(
        ge=0,
        le=1,
        description="The same for the trend strength, by number of periods.",
    )


class SeasonalDecompositionOutput(TimeSeriesOutput):
    period: int | None = Field(
        description="The period used for every column; None when none was found."
    )
    period_source: str = Field(
        description="`configured`, or the frequency it was derived from."
    )
    min_strength: float
    columns: dict[str, ColumnDecomposition] = Field(
        description="Every column that was decomposed, by name. The component "
        "series are not kept."
    )


@check(
    id="seasonal_decomposition",
    version="1.1.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric", "datetime"),
    dimensions=[Dimension.TEMPORAL],
    params=SeasonalDecompositionParams,
    output=SeasonalDecompositionOutput,
    requires=Requires(profile=["counts"], params=["time_index"]),
    cost=Cost.MODERATE,
    extras=["timeseries"],
    packs=["core"],
    tags=["time-series", "seasonality", "trend", "stl"],
)
def seasonal_decomposition(
    ctx: CheckContext, params: SeasonalDecompositionParams
) -> CheckOutput:
    """Split each numeric column into trend, seasonal and remainder by STL.

    The columns are resampled to the frequency first (see ``checks.timeseries``) and
    the most recent ``max_periods`` periods are used. The output keeps the strengths,
    not the component series.
    """
    prepared = prepare(ctx, params, min_periods=2 * params.min_cycles)
    if isinstance(prepared, CheckOutput):
        return prepared
    period, source = _period(params, prepared)
    if period is None:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_seasonal_period",
            f"The frequency {prepared.step.alias!r} implies no seasonal period, so "
            "there is no cycle to decompose.",
            severity=Severity.INFO,
            hint="Set `period`, the number of periods in one cycle.",
        )
    needed = params.min_cycles * period
    columns: dict[str, ColumnDecomposition] = {}
    skipped = dict(prepared.skipped)
    for name, series in prepared.series.items():
        ctx.checkpoint()
        if series.values.size < needed:
            skipped[name] = (
                f"{series.values.size} periods, fewer than {params.min_cycles} "
                f"cycles of {period}"
            )
            continue
        result = _decompose(series.values, period, params, prepared.info(name))
        if result is None:
            skipped[name] = "the decomposition did not converge to finite values"
            continue
        columns[name] = result
        _report(ctx, name, result, params, prepared)
    if not columns:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_series_long_enough",
            f"No series has {needed} or more periods, {params.min_cycles} cycles "
            f"of {period}.",
            severity=Severity.INFO,
        )
    return ctx.output(
        prepared.build(
            SeasonalDecompositionOutput,
            skipped=skipped,
            period=period,
            period_source=source,
            min_strength=params.min_strength,
            columns=columns,
        )
    )


def _period(
    params: SeasonalDecompositionParams, prepared: Prepared
) -> tuple[int | None, str]:
    if params.period is not None:
        return params.period, "configured"
    step = prepared.step
    cycle = _CYCLE.get(step.unit)
    if cycle is None or cycle % step.count or cycle // step.count < 2:  # noqa: PLR2004
        return None, step.alias
    return cycle // step.count, step.alias


def _decompose(
    values: np.ndarray,
    period: int,
    params: SeasonalDecompositionParams,
    info: SeriesInfo,
) -> ColumnDecomposition | None:
    from statsmodels.tsa.seasonal import STL  # the `timeseries` extra

    fit = STL(values, period=period, robust=params.robust).fit()
    trend, seasonal, remainder = fit.trend, fit.seasonal, fit.resid
    with np.errstate(all="ignore"):
        variance = float(np.var(remainder, ddof=1))
        detrended = float(np.var(trend + remainder, ddof=1))
        deseasoned = float(np.var(seasonal + remainder, ddof=1))
        total = float(np.var(values, ddof=1))
        share = variance / total if total > 0 else 0.0
    if not all(np.isfinite([variance, detrended, deseasoned, share])):
        return None
    return ColumnDecomposition(
        **info.model_dump(),
        period=period,
        robust=params.robust,
        trend_strength=_strength(variance, detrended),
        seasonal_strength=_strength(variance, deseasoned),
        remainder_share=share,
        seasonal_floor=_noise_floor(_SEASONAL_FLOOR, values.size / period),
        trend_floor=_noise_floor(_TREND_FLOOR, values.size),
    )


def _noise_floor(fit: tuple[float, float], size: float) -> float:
    """The strength white noise reaches in 1% of series, for a series of this size."""
    return min(1.0, fit[0] + fit[1] / np.sqrt(size))


def _strength(remainder: float, with_remainder: float) -> float:
    if with_remainder <= 0:
        return 0.0
    return min(1.0, max(0.0, 1.0 - remainder / with_remainder))


def _label(strength: float) -> str:
    if strength >= _STRONG:
        return "strong"
    if strength >= _MODERATE:
        return "moderate"
    return "weak"


def _report(
    ctx: CheckContext,
    name: str,
    result: ColumnDecomposition,
    params: SeasonalDecompositionParams,
    prepared: Prepared,
) -> None:
    for code, kind, strength, floor in (
        ("seasonal", "seasonality", result.seasonal_strength, result.seasonal_floor),
        ("trend", "trend", result.trend_strength, result.trend_floor),
    ):
        if strength < max(params.min_strength, floor):
            continue
        label = _label(strength)
        metrics: dict[str, object] = {
            "period": result.period,
            "robust": result.robust,
            "trend_strength": result.trend_strength,
            "seasonal_strength": result.seasonal_strength,
            "remainder_share": result.remainder_share,
            "min_strength": params.min_strength,
            "noise_floor": floor,
        }
        ctx.finding(
            code=code,
            subject=Subject.column(name),
            severity=Severity.INFO,
            confidence=Confidence.MODERATE,
            title=f"{label.capitalize()} {kind} (strength {strength:.2f})",
            message=(
                f"STL of {name!r} against {prepared.index!r}, as {result.periods:,} "
                f"{prepared.step.alias} periods with a cycle of {result.period}: "
                f"the {kind} has {label} strength {strength:.2f} (trend "
                f"{result.trend_strength:.2f}, seasonal {result.seasonal_strength:.2f}"
                f", remainder {result.remainder_share:.0%} of the variance). "
                + (
                    f"Model the cycle of {result.period} periods, for example with "
                    "a seasonal difference or calendar features, and split by time "
                    "so every cycle is represented in training."
                    if code == "seasonal"
                    else "Detrend or difference the series before models that "
                    "assume a stable level, and do not split it at random."
                )
            ),
            evidence=prepared.evidence(ctx, name, metrics, "stl"),
        )
