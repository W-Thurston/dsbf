# dsbf/profile/types.py
"""Semantic types, inferred from a sample of each column.

A semantic type says how analysis should treat a column (``numeric``, ``categorical``,
``boolean``, ``datetime``, ``text``, ``identifier`` or ``unknown``), where the storage
dtype says only how it is held.

Facts Polars answers in one native pass are read from the whole column: the null
count, the exact distinct count, the median. Parsing and string-shape rules read a
seeded sample of at most ``sample_rows`` non-null values, so their cost does not grow
with the row count. A column that passes the date test on the sample is then parsed
whole to confirm it.

The rules, in order:

1. No value (all null, or float NaN only) is ``unknown``, whatever dtype held the nulls.
2. A dtype with no profile (nested, binary, duration, time) is ``unknown``.
3. ``Boolean`` is ``boolean``; ``Date`` and ``Datetime`` are ``datetime``.
4. A numeric dtype is ``categorical`` with 2 distinct values, or with at most 20 that
   are under 5% of the values; ``datetime`` if the median is a Unix timestamp; else
   ``numeric``.
5. Text (``String``, ``Categorical``, ``Enum``) is ``datetime`` if one known format
   parses over 90% of the sample and of the column; else ``categorical`` if it is
   mostly period labels such as ``FY2024``; else ``identifier`` if it is mostly
   hex-like or over 90% distinct; else ``text`` if the mean length is over 30; else
   ``categorical``.
"""

from dataclasses import dataclass
from typing import Literal, cast

import polars as pl

from dsbf.data.dataset import Dataset
from dsbf.model.profile import SemanticType
from dsbf.model.vocabulary import Confidence

TYPE_SAMPLE_ROWS = 1000
_SAMPLE_SEED = 0

_MIN_CONFIDENT_ROWS = 10
_DATE_SHARE = 0.9  # of the values a format must parse, in the sample and the column
_FIRM_DATE_SHARE = 0.99
_ID_SHARE = 0.8
_ID_UNIQUE_SHARE = 0.9
_PERIOD_LABEL_SHARE = 0.5
_PERIOD_LABEL_MAX_DISTINCT = 100
_TEXT_MEAN_LENGTH = 30
_CATEGORICAL_MAX_DISTINCT = 20
_CATEGORICAL_UNIQUE_SHARE = 0.05
_MIN_EPOCH_ROWS = 5
_EPOCH_RANGES: dict[Literal["s", "ms"], tuple[int, int]] = {
    "s": (1_000_000_000, 2_000_000_000),  # 2001 to 2033
    "ms": (1_000_000_000_000, 2_000_000_000_000),
}

# Fixed list, tried in order; the earlier format wins a tie, so a slash date whose
# days are all under 13 reads month-first, as pandas assumes. `%Y%m%d` is left out on
# purpose: eight-digit identifiers parse as dates under it.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S%.fZ",
    "%Y-%m-%dT%H:%M:%S%.f%z",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f%z",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%b %d %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%b-%Y",
)

# Fiscal-year, quarter and short-year labels look numeric but are categories.
_PERIOD_LABEL = "|".join(
    f"(?:{pattern})"
    for pattern in (
        r"(?i)^FY\d{2,4}$",
        r"^\d{4}$",
        r"^'\d{2}$",
        r"^\d{4}[-/]\d{2,4}$",
        r"^'\d{2}[-/]\d{2}$",
        r"^\d{2}/\d{2}$",
        r"(?i)^Q[1-4]\s?\d{2,4}$",
    )
)
_HEX_LIKE = r"^[A-Fa-f0-9\-]{8,}$"

_UNPROFILED_DTYPES = (
    pl.Null,
    pl.Binary,
    pl.List,
    pl.Array,
    pl.Struct,
    pl.Object,
    pl.Duration,
    pl.Time,
)
_TEXT_DTYPES = (pl.String, pl.Categorical, pl.Enum)


@dataclass(frozen=True, slots=True)
class ColumnType:
    """The semantic type of one column, and what it was inferred from."""

    semantic_type: SemanticType
    confidence: Confidence
    physical_dtype: str
    reason: str
    sample_rows: int | None = None
    """Non-null values the parsing and string rules read; None when the dtype or
    whole-column statistics decided."""
    datetime_format: str | None = None
    """For a text column read as dates, the strptime format that parses it."""
    epoch_unit: Literal["s", "ms"] | None = None
    """For a numeric column read as dates, the unit of its Unix timestamps."""


def infer_column_types(
    dataset: Dataset, *, sample_rows: int = TYPE_SAMPLE_ROWS
) -> dict[str, ColumnType]:
    """The semantic type of every column of ``dataset``, in column order."""
    return {
        name: infer_column_type(dataset.frame[name], sample_rows=sample_rows)
        for name in dataset.columns
    }


def infer_column_type(
    series: pl.Series, *, sample_rows: int = TYPE_SAMPLE_ROWS
) -> ColumnType:
    """The semantic type of one column.

    Text rules read at most ``sample_rows`` non-null values, drawn with a fixed seed,
    so the answer for the same data does not change between runs.

    Raises:
        ValueError: ``sample_rows`` is not a positive integer.
    """
    if isinstance(sample_rows, bool) or not isinstance(sample_rows, int):
        raise ValueError(f"sample_rows must be a positive integer, not {sample_rows!r}")
    if sample_rows < 1:
        raise ValueError(f"sample_rows must be a positive integer, not {sample_rows!r}")
    dtype = series.dtype
    present = _present(series)
    if present.is_empty():
        return _unknown(dtype, "no values")
    if dtype in _UNPROFILED_DTYPES:
        return _unknown(dtype, f"{dtype} has no profile")
    if dtype == pl.Boolean:
        return _typed("boolean", Confidence.HIGH, dtype, "boolean dtype", present)
    if dtype in (pl.Date, pl.Datetime):
        return _typed("datetime", Confidence.HIGH, dtype, "date dtype", present)
    if dtype.is_numeric():
        return _numeric(present)
    if dtype in _TEXT_DTYPES:
        return _text(present, sample_rows)
    return _unknown(dtype, f"{dtype} has no profile")


def _present(series: pl.Series) -> pl.Series:
    """The values that are neither null nor float NaN."""
    values = series.drop_nulls()
    return values.drop_nans() if series.dtype.is_float() else values


def _typed(
    semantic_type: SemanticType,
    confidence: Confidence,
    dtype: pl.DataType,
    reason: str,
    present: pl.Series,
    **fields,
) -> ColumnType:
    """A result, with confidence capped when there are too few values to trust."""
    if present.len() < _MIN_CONFIDENT_ROWS:
        confidence = min(confidence, Confidence.LOW)
    return ColumnType(semantic_type, confidence, str(dtype), reason, **fields)


def _unknown(dtype: pl.DataType, reason: str) -> ColumnType:
    return ColumnType("unknown", Confidence.LOW, str(dtype), reason)


def _numeric(present: pl.Series) -> ColumnType:
    dtype = present.dtype
    count = present.len()
    distinct = present.n_unique()
    if distinct == 2:  # noqa: PLR2004
        return _typed(
            "categorical", Confidence.MODERATE, dtype, "two distinct values", present
        )
    if (
        distinct <= _CATEGORICAL_MAX_DISTINCT
        and distinct / count < _CATEGORICAL_UNIQUE_SHARE
    ):
        return _typed(
            "categorical", Confidence.MODERATE, dtype, "few distinct values", present
        )
    unit = _epoch_unit(present)
    if unit is not None:
        return _typed(
            "datetime",
            Confidence.MODERATE,
            dtype,
            "Unix timestamps",
            present,
            epoch_unit=unit,
        )
    return _typed("numeric", Confidence.HIGH, dtype, "numeric dtype", present)


def _epoch_unit(present: pl.Series) -> Literal["s", "ms"] | None:
    if present.len() < _MIN_EPOCH_ROWS:
        return None
    median = cast("float | None", present.cast(pl.Float64).median())
    if median is None:
        return None
    for unit, (low, high) in _EPOCH_RANGES.items():
        if low < median < high:
            return unit
    return None


def _text(present: pl.Series, sample_rows: int) -> ColumnType:
    dtype = present.dtype
    values = present if dtype == pl.String else present.cast(pl.String)
    count = values.len()
    distinct = values.n_unique()
    if count > sample_rows:
        sample = values.sample(sample_rows, seed=_SAMPLE_SEED).str.strip_chars()
    else:
        sample = values.str.strip_chars()

    def result(
        semantic_type: SemanticType,
        confidence: Confidence,
        reason: str,
        **fields,
    ) -> ColumnType:
        return _typed(
            semantic_type,
            confidence,
            dtype,
            reason,
            present,
            sample_rows=sample.len(),
            **fields,
        )

    found = _date_format(sample, values, count)
    if found is not None:
        fmt, share = found
        firm = share > _FIRM_DATE_SHARE
        return result(
            "datetime",
            Confidence.HIGH if firm else Confidence.MODERATE,
            "dates in a known format",
            datetime_format=fmt,
        )
    if (
        distinct < _PERIOD_LABEL_MAX_DISTINCT
        and _share(sample.str.contains(_PERIOD_LABEL)) > _PERIOD_LABEL_SHARE
    ):
        return result("categorical", Confidence.MODERATE, "period labels")
    if (
        _share(sample.str.contains(_HEX_LIKE)) > _ID_SHARE
        or distinct / count > _ID_UNIQUE_SHARE
    ):
        return result(
            "identifier", Confidence.MODERATE, "hex-like or nearly all distinct"
        )
    if cast("float", sample.str.len_chars().mean()) > _TEXT_MEAN_LENGTH:
        return result("text", Confidence.MODERATE, "long strings")
    return result("categorical", Confidence.MODERATE, "repeated strings")


def _share(matches: pl.Series) -> float:
    return cast("int", matches.sum()) / matches.len()


def _parse(strings: pl.Series, fmt: str) -> pl.Series:
    return strings.str.to_datetime(fmt, time_unit="us", strict=False)


def _date_format(
    sample: pl.Series, values: pl.Series, count: int
) -> tuple[str, float] | None:
    """The format that parses the sample, and the share of the column it parses.

    The best format on the sample is the candidate; it must parse over 90% of the
    sample, and then over 90% of the whole column.
    """
    best_format, best_hits = None, 0
    for fmt in _DATE_FORMATS:
        hits = sample.len() - _parse(sample, fmt).null_count()
        if hits > best_hits:
            best_format, best_hits = fmt, hits
    if best_format is None or best_hits / sample.len() <= _DATE_SHARE:
        return None
    if sample.len() == count:
        return best_format, best_hits / count
    whole = _parse(values.str.strip_chars(), best_format)
    share = (count - whole.null_count()) / count
    return (best_format, share) if share > _DATE_SHARE else None
