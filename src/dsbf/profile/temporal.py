# dsbf/profile/temporal.py
"""Range, time zone, frequency and gaps of a datetime column.

Instants come from the column as stored (Date, Datetime), from a text column parsed
with the format type inference found, or from a numeric column read as Unix
timestamps in the unit it found. Frequency and gaps are worked out on the sorted
distinct instants, so repeated timestamps are neither gaps nor steps.

The frequency is the most common step between consecutive instants, and only counts
when at least half the steps are that long. A step is a calendar step when every
instant is a month start (or every one a month end) at one time of day: the spacing
is then counted in months, because months differ in length. Anything else is a fixed
span of seconds. The alias names the spacing; for weeks, months and years it does not
record which weekday or month the series is anchored to.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import numpy as np
import polars as pl

from dsbf.profile.types import ColumnType, parse_dates

_MIN_INSTANTS_FOR_FREQUENCY = 3
_DOMINANT_STEP_SHARE = 0.5
_GAP_FACTOR = 1.5
_US = {
    "D": 86_400_000_000,
    "h": 3_600_000_000,
    "min": 60_000_000,
    "s": 1_000_000,
    "ms": 1_000,
}


@dataclass(frozen=True, slots=True)
class DatetimeStats:
    """The datetime fields of a column profile; None where they cannot be known."""

    min: datetime | None = None
    max: datetime | None = None
    timezone: str | None = None
    inferred_frequency: str | None = None
    n_gaps: int | None = None
    largest_gap_seconds: float | None = None


def datetime_stats(series: pl.Series, column_type: ColumnType) -> DatetimeStats:
    """The range, time zone, frequency and gaps of a column typed as datetime."""
    instants = _instants(series, column_type)
    if instants is None:
        return DatetimeStats()
    instants = instants.drop_nulls()
    if instants.is_empty():
        return DatetimeStats()
    timezone = getattr(instants.dtype, "time_zone", None)
    low, high = _bounds(instants)
    distinct = instants.unique().sort()
    gaps = _gaps(distinct)
    return DatetimeStats(min=low, max=high, timezone=timezone, **gaps)


def _instants(series: pl.Series, column_type: ColumnType) -> pl.Series | None:
    dtype = series.dtype
    if dtype == pl.Date:
        return series.cast(pl.Datetime("us"))
    if isinstance(dtype, pl.Datetime):
        return series.dt.cast_time_unit("us")
    if dtype in (pl.String, pl.Categorical, pl.Enum) and column_type.datetime_format:
        text = series.cast(pl.String).str.strip_chars()
        parsed = parse_dates(text, column_type.datetime_format)
        if column_type.datetime_format.endswith("Z"):
            parsed = parsed.dt.replace_time_zone("UTC")
        return parsed
    if dtype.is_numeric() and column_type.epoch_unit:
        scale = 1_000_000 if column_type.epoch_unit == "s" else 1_000
        micros = (series.cast(pl.Float64) * scale).cast(pl.Int64, strict=False)
        return micros.cast(pl.Datetime("us"))
    return None


def _bounds(instants: pl.Series) -> tuple[datetime | None, datetime | None]:
    # Python's datetime stops at year 9999; a range past it is not representable.
    try:
        return (
            cast("datetime | None", instants.min()),
            cast("datetime | None", instants.max()),
        )
    except (ValueError, OverflowError):
        return None, None


def _gaps(distinct: pl.Series) -> dict[str, Any]:
    if distinct.len() < 2:  # noqa: PLR2004
        return {}
    micros = distinct.dt.epoch("us").to_numpy()
    steps = np.diff(micros)
    result: dict[str, Any] = {"largest_gap_seconds": float(steps.max()) / 1e6}
    if distinct.len() < _MIN_INSTANTS_FOR_FREQUENCY:
        return result
    calendar = _calendar_steps(distinct)
    if calendar is not None:
        months, suffix = calendar
        step, share = _dominant(months)
        if share >= _DOMINANT_STEP_SHARE:
            result["inferred_frequency"] = _month_alias(step, suffix)
            result["n_gaps"] = int((months > _GAP_FACTOR * step).sum())
        return result
    step, share = _dominant(steps)
    if share >= _DOMINANT_STEP_SHARE:
        result["inferred_frequency"] = _span_alias(step)
        result["n_gaps"] = int((steps > _GAP_FACTOR * step).sum())
    return result


def _dominant(steps: np.ndarray) -> tuple[int, float]:
    """The most common step (the shortest on a tie) and the share of steps it is."""
    counts = Counter(steps.tolist())
    step, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
    return int(step), count / steps.size


def _calendar_steps(distinct: pl.Series) -> tuple[np.ndarray, str] | None:
    """Months between consecutive instants, if they are all month starts or ends."""
    if distinct.dt.time().n_unique() != 1:
        return None
    if distinct.dt.day().max() == 1 and distinct.dt.day().min() == 1:
        suffix = "MS"
    elif (distinct.dt.offset_by("1d").dt.day() == 1).all():
        suffix = "ME"
    else:
        return None
    index = distinct.dt.year().cast(pl.Int64) * 12 + distinct.dt.month().cast(pl.Int64)
    return np.diff(index.to_numpy()), suffix


def _month_alias(months: int, suffix: str) -> str:
    return suffix if months == 1 else f"{months}{suffix}"


def _span_alias(micros: int) -> str:
    """A pandas-style alias for a fixed span: ``D``, ``W``, ``2D``, ``15min``, ``h``."""
    for unit in ("D", "h", "min", "s", "ms"):
        size = _US[unit]
        if micros % size == 0:
            count = micros // size
            if unit == "D" and count % 7 == 0:  # noqa: PLR2004
                weeks = count // 7
                return "W" if weeks == 1 else f"{weeks}W"
            return unit if count == 1 else f"{count}{unit}"
    return "us" if micros == 1 else f"{micros}us"
