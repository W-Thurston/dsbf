# dsbf/checks/timeseries.py
"""What the time-series checks share: the index, the frequency and the resampling.

This module holds no check. ``stationarity``, ``autocorrelation`` and
``seasonal_decomposition`` call ``prepare`` and get one regular series per numeric
column, so they test the same thing and say how it was made.

The tests those checks run assume equally spaced observations. The rows of a file are
not: they repeat a timestamp, skip periods and, in a long log, number millions. So
the series is built in four steps. The rows with a usable timestamp are binned to the
frequency (the one configured, else the one the profile found). The values of a bin
are aggregated. The empty bins between the first and last value are filled by linear
interpolation, up to a share of the series. And only the most recent ``max_periods``
periods are kept, which bounds the cost and keeps a huge series from rejecting every
null hypothesis over negligible deviations.
"""

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from pydantic import Field, field_validator

from dsbf.engine import CheckContext, CheckOutput, CheckOutputModel
from dsbf.model import (
    CheckParams,
    CheckStatus,
    Contract,
    DatetimeProfile,
    Evidence,
    NumericProfile,
    Severity,
)
from dsbf.profiling.temporal import instants

Aggregate = Literal["mean", "median", "sum", "last"]

# Why a column has no series.
OVERFLOW = "values too large to aggregate"

_US_PER_UNIT = {
    "D": 86_400_000_000,
    "h": 3_600_000_000,
    "min": 60_000_000,
    "s": 1_000_000,
    "ms": 1_000,
    "us": 1,
}
_MONTHS_PER_UNIT = {"MS": 1, "ME": 1, "QS": 3, "QE": 3, "YS": 12, "YE": 12}
_ALIAS = re.compile(r"(?P<count>\d*)(?P<unit>[A-Za-z]+)")
_MICROS = 1_000_000
# The largest magnitude of a microsecond count that survives the arithmetic below.
_MAX_MICROS = 2**62


class TimeSeriesParams(CheckParams):
    """What every time-series check reads: the index and how to resample to it."""

    time_index: str | None = Field(
        None,
        description="The datetime column that orders the rows. "
        "`dsbf.profile(time_index=...)` sets it; without one the check is not "
        "applicable.",
    )
    frequency: str | None = Field(
        None,
        description="The spacing of the series, as an offset alias: a count and a "
        "unit (`D`, `h`, `min`, `s`, `ms`, `W`: fixed spans; `MS`, `ME`, `QS`, "
        "`QE`, `YS`, `YE`: calendar months), such as `15min` or `2W`. Unset, the "
        "spacing the profile found in the index is used.",
    )
    aggregate: Aggregate = Field(
        "mean",
        description="How the values of one period are combined, which is what "
        "repeated timestamps and rows finer than the frequency need.",
    )
    max_periods: int = Field(
        2000,
        ge=30,
        le=100_000,
        description="Most periods used: the most recent ones. Tests on tens of "
        "thousands of points reject trivial deviations, and the cost of the lag "
        "regressions grows with them.",
    )
    max_fill_share: float = Field(
        0.2,
        ge=0,
        le=0.9,
        description="Largest share of a series' periods that may be empty and "
        "interpolated; a column with more has no series.",
    )
    value_columns: tuple[str, ...] = Field(
        (),
        description="The numeric columns to analyse; empty is every numeric column.",
    )

    @field_validator("frequency")
    @classmethod
    def _valid_frequency(cls, value: str | None) -> str | None:
        if value is not None:
            parse_frequency(value)
        return value


class SeriesInfo(Contract):
    """How one column's series was made; every column's result starts with it."""

    periods: int = Field(ge=1, description="Periods in the series: its length.")
    periods_filled: int = Field(
        ge=0, description="Empty periods filled by linear interpolation."
    )
    rows_used: int = Field(ge=1, description="Rows whose value entered the series.")
    series_start: str = Field(description="Start of the first period, UTC.")


class TimeSeriesOutput(CheckOutputModel):
    """What every time-series check's output states of the series it analysed."""

    index: str = Field(description="The datetime column that ordered the rows.")
    frequency: str = Field(description="The period of every series, as an alias.")
    frequency_source: Literal["configured", "profile"]
    aggregate: Aggregate = Field(description="How the values of one period combine.")
    max_periods: int = Field(ge=1, description="Most recent periods used.")
    periods_total: int = Field(
        ge=1, description="Periods the index spans, before the cut to `max_periods`."
    )
    truncated: bool = Field(description="The index spans more than `max_periods`.")
    timezone: str | None = Field(
        description="Time zone of the index; instants are read as UTC."
    )
    skipped: dict[str, str] = Field(description="Columns with no result, and why.")


@dataclass(frozen=True, slots=True)
class Step:
    """One period of a frequency: a span of microseconds or a count of months.

    ``count`` units of ``unit`` make the period (``15min`` is 15 of ``min``).
    """

    alias: str
    micros: int | None
    months: int | None
    unit: str
    count: int


@dataclass(frozen=True, slots=True)
class Series:
    """One column as a regular series, and how it was made."""

    values: np.ndarray
    rows: int
    periods_filled: int
    first_period: int
    start_micros: int


@dataclass(frozen=True, slots=True)
class Prepared:
    """The series of every usable column, and what they share."""

    index: str
    step: Step
    frequency_source: Literal["configured", "profile"]
    aggregate: Aggregate
    max_periods: int
    periods_total: int
    timezone: str | None
    series: dict[str, Series]
    skipped: dict[str, str]

    @property
    def truncated(self) -> bool:
        return self.periods_total > self.max_periods

    def build[T: TimeSeriesOutput](self, model: type[T], **fields: object) -> T:
        """The check's output: ``fields`` and what every time-series output states."""
        return model.model_validate({**self._header(), **fields})

    def _header(self) -> dict[str, object]:
        return {
            "index": self.index,
            "frequency": self.step.alias,
            "frequency_source": self.frequency_source,
            "aggregate": self.aggregate,
            "max_periods": self.max_periods,
            "periods_total": self.periods_total,
            "truncated": self.truncated,
            "timezone": self.timezone,
            "skipped": self.skipped,
        }

    def info(self, name: str) -> SeriesInfo:
        """How the series of ``name`` was made."""
        series = self.series[name]
        return SeriesInfo(
            periods=int(series.values.size),
            periods_filled=series.periods_filled,
            rows_used=series.rows,
            series_start=_start(series),
        )

    def metrics(self, name: str) -> dict[str, object]:
        """What every finding about ``name`` states of how its series was made."""
        series = self.series[name]
        return {
            "index": self.index,
            "frequency": self.step.alias,
            "frequency_source": self.frequency_source,
            "aggregate": self.aggregate,
            "periods": int(series.values.size),
            "periods_total": self.periods_total,
            "periods_filled": series.periods_filled,
            "series_start": _start(series),
            "rows_used": series.rows,
            "max_periods": self.max_periods,
            "truncated": self.truncated,
        }

    def evidence(
        self,
        ctx: CheckContext,
        name: str,
        metrics: dict[str, object],
        method: str,
    ) -> Evidence:
        """The evidence of a finding about ``name``: its series plus ``metrics``."""
        periods = int(self.series[name].values.size)
        return Evidence(
            metrics={**self.metrics(name), **metrics},  # type: ignore[arg-type]
            n_used=periods,
            # Filled periods are counted in the series but not in the rows.
            n_total=max(ctx.profile.n_rows, periods),
            method=method,
        )


def _start(series: Series) -> str:
    return str(np.datetime64(series.start_micros, "us").astype("datetime64[s]"))


def parse_frequency(alias: str) -> Step:
    """The period an offset alias such as ``D``, ``2W`` or ``15min`` names.

    Raises:
        ValueError: the alias is not a count and a supported unit.
    """
    found = _ALIAS.fullmatch(alias.strip())
    unit = found["unit"] if found else ""
    count = int(found["count"] or 1) if found else 0
    if found is None or count < 1:
        raise ValueError(f"{alias!r} is not an offset alias such as 'D' or '15min'")
    if unit == "W":
        return Step(alias, 7 * _US_PER_UNIT["D"] * count, None, unit, count)
    if unit in _US_PER_UNIT:
        return Step(alias, _US_PER_UNIT[unit] * count, None, unit, count)
    if unit in _MONTHS_PER_UNIT:
        return Step(alias, None, _MONTHS_PER_UNIT[unit] * count, unit, count)
    supported = ", ".join(("W", *_US_PER_UNIT, *_MONTHS_PER_UNIT))
    raise ValueError(f"{alias!r} has the unit {unit!r}; supported units: {supported}")


def read_instants(
    raw: np.ndarray, column: DatetimeProfile
) -> tuple[np.ndarray, np.ndarray]:
    """A datetime column's instants as microseconds, and which rows have one.

    ``raw`` is the column read with ``dropna=False``. It is read as the profile read
    it: with its ``format`` or ``epoch_unit``. A row whose value is missing or does
    not read, or lies outside the range the arithmetic holds, has no instant.
    Instants are UTC.
    """
    series = pl.Series("t", raw.tolist() if raw.dtype == object else raw)
    read = instants(series, format=column.format, epoch_unit=column.epoch_unit)
    if read is None:
        return np.zeros(raw.size, dtype=np.int64), np.zeros(raw.size, dtype=bool)
    if getattr(read.dtype, "time_zone", None) is not None:
        read = read.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    present = read.is_not_null().to_numpy()
    micros = read.dt.epoch("us").fill_null(0).to_numpy()
    present &= np.abs(micros) < _MAX_MICROS
    return micros, present


def _bins(micros: np.ndarray, step: Step) -> np.ndarray:
    """The period each (sorted) instant falls in, counted from the first."""
    if step.micros is not None:
        return (micros - micros[0]) // step.micros
    months = micros.astype("datetime64[us]").astype("datetime64[M]").astype(np.int64)
    assert step.months is not None
    return (months - months[0]) // step.months


def _period_start(micros: int, period: int, step: Step) -> int:
    """The first instant of period ``period``, counted from the instant ``micros``."""
    if step.micros is not None:
        return micros + period * step.micros
    assert step.months is not None
    first = np.datetime64(micros, "us").astype("datetime64[M]")
    return int(
        (first + np.timedelta64(period * step.months, "M"))
        .astype("datetime64[us]")
        .astype(np.int64)
    )


def _aggregate(frame: pl.DataFrame, how: Aggregate) -> tuple[np.ndarray, np.ndarray]:
    expression = {
        "mean": pl.col("v").mean(),
        "median": pl.col("v").median(),
        "sum": pl.col("v").sum(),
        "last": pl.col("v").last(),
    }[how]
    grouped = frame.group_by("b", maintain_order=True).agg(expression)
    return grouped["b"].to_numpy(), grouped["v"].to_numpy()


def _stopped(
    ctx: CheckContext, code: str, message: str, hint: str | None = None
) -> CheckOutput:
    return ctx.stop(
        CheckStatus.NOT_APPLICABLE,
        code,
        message,
        severity=Severity.INFO,
        hint=hint,
    )


def prepare(
    ctx: CheckContext, params: TimeSeriesParams, *, min_periods: int
) -> Prepared | CheckOutput:
    """One regular series per numeric column, or the output that ends the check.

    A series needs at least ``min_periods`` periods; a column with fewer, too many
    empty periods or no variation is in ``skipped`` with the reason. The check ends
    ``not_applicable`` when there is no usable index or no usable series.
    """
    name = params.time_index
    if name is None:
        return _stopped(
            ctx,
            "no_time_index",
            "No time index is set, so there is no time axis to analyse.",
            "Pass time_index=... to dsbf.profile(), or set the `time_index` parameter.",
        )
    column = ctx.profile.column(name) if name in ctx.columns else None
    if not isinstance(column, DatetimeProfile):
        return _stopped(
            ctx,
            "time_index_not_a_datetime",
            f"The time index {name!r} is not a column the profile typed as "
            "datetime, so it cannot order the rows.",
        )
    sampling = ctx.sampling
    if sampling is not None and sampling.strategy != "head":
        ctx.note(
            "sampled_series",
            f"The data is a {sampling.strategy} sample of {sampling.sampled_rows:,} of "
            f"{sampling.original_rows:,} rows, so periods of the series may be empty "
            "and interpolated, and each period aggregates fewer rows.",
            hint="Profile the whole data, or sample the head of an ordered file.",
        )
    source = "configured" if params.frequency else "profile"
    alias = params.frequency or column.inferred_frequency
    if alias is None:
        return _stopped(
            ctx,
            "no_regular_frequency",
            f"The time index {name!r} has no dominant spacing between its values, "
            "so there is no regular series to build.",
            "Set `frequency` (for example 'D' or '15min') to resample to it.",
        )
    step = parse_frequency(alias)
    micros, present = read_instants(ctx.values(name, dropna=False), column)
    if not present.any():
        return _stopped(
            ctx, "no_instants", f"No value of {name!r} reads as a date or time."
        )

    rows = np.flatnonzero(present)
    order = rows[np.argsort(micros[rows], kind="stable")]
    origin = int(micros[order[0]])
    bins = _bins(micros[order], step)
    periods_total = int(bins[-1] - bins[0]) + 1
    kept = int(np.searchsorted(bins, bins[-1] - params.max_periods + 1))
    order, bins = order[kept:], bins[kept:]

    wanted = set(params.value_columns)
    skipped: dict[str, str] = {
        c: "not a numeric column of the data"
        for c in params.value_columns
        if c not in ctx.columns or c == name
    }
    series: dict[str, Series] = {}
    for column_name in ctx.columns:
        profile = ctx.profile.column(column_name)
        if not isinstance(profile, NumericProfile):
            continue
        if wanted and column_name not in wanted:
            continue
        result = _series(
            ctx.values(column_name, dropna=False)[order],
            bins,
            origin,
            step,
            params,
            min_periods,
        )
        if isinstance(result, str):
            skipped[column_name] = result
        else:
            series[column_name] = result
    if not series:
        return _stopped(
            ctx,
            "no_series",
            f"No numeric column has a usable series of {min_periods} or more "
            f"{step.alias} periods against {name!r}.",
        )
    return Prepared(
        index=name,
        step=step,
        frequency_source=source,
        aggregate=params.aggregate,
        max_periods=params.max_periods,
        periods_total=periods_total,
        timezone=column.timezone,
        series=series,
        skipped=skipped,
    )


def _series(
    values: np.ndarray,
    bins: np.ndarray,
    origin: int,
    step: Step,
    params: TimeSeriesParams,
    min_periods: int,
) -> Series | str:
    """The column's regular series over ``bins``, or the reason it has none."""
    numbers = values.astype(np.float64, copy=False)
    finite = np.isfinite(numbers)
    if not finite.any():
        return "no finite values"
    frame = pl.DataFrame({"b": bins[finite], "v": numbers[finite]})
    with np.errstate(all="ignore"):
        keys, aggregated = _aggregate(frame, params.aggregate)
    if not np.isfinite(aggregated).all():
        return OVERFLOW
    first = int(keys[0])
    length = int(keys[-1]) - first + 1
    if length < min_periods:
        return f"fewer than {min_periods} periods"
    regular = np.full(length, np.nan)
    regular[keys - first] = aggregated
    empty = np.isnan(regular)
    filled = int(empty.sum())
    if filled > params.max_fill_share * length:
        return (
            f"{filled / length:.0%} of its periods have no value "
            f"(at most {params.max_fill_share:.0%} are filled)"
        )
    if filled:
        known = np.flatnonzero(~empty)
        regular[empty] = np.interp(np.flatnonzero(empty), known, regular[known])
    if np.ptp(regular) == 0:
        return "constant"
    return Series(
        values=regular,
        rows=int(finite.sum()),
        periods_filled=filled,
        first_period=first,
        start_micros=_period_start(origin, first, step),
    )
