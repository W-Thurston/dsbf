# dsbf/checks/strings.py
"""What the string checks share: which columns hold strings, and how to read them.

This module holds no check. ``string_anomalies``, ``encoded_columns``,
``mixed_types`` and ``format_violations`` import it, so they read a column's distinct
values the same way and agree on which characters are invisible.
"""

import numpy as np
import polars as pl

from dsbf.model import ColumnProfile

# The semantic types a string can have. A key (`identifier`) and prose (`text`) hold
# the same kinds of mistake as a label (`categorical`) does.
STRING_TYPES = ("categorical", "text", "identifier")

# Zero-width and joining characters, the soft hyphen, the no-break space, the line and
# paragraph separators, the bidirectional controls, and the C0 and C1 controls other
# than tab, line feed, carriage return and NUL (`text_encoding` reports NUL).
INVISIBLE = (
    r"[­ ​-‍  ‪-‮⁠⁦-⁩﻿"
    r"\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


def holds_strings(column: ColumnProfile) -> bool:
    """Whether the column is stored as strings (an integer code has no string shape)."""
    dtype = column.physical_dtype
    return dtype in ("String", "Utf8") or dtype.startswith(("Categorical", "Enum"))


def distinct_values(values: np.ndarray) -> pl.DataFrame:
    """The distinct strings of a column and how many rows have each.

    ``values`` is a column read with ``dropna=True``. The result has the columns
    ``value`` (String) and ``count`` (Int64), in no particular order.
    """
    series = pl.Series("value", values, dtype=pl.String)
    return series.value_counts().with_columns(pl.col("count").cast(pl.Int64))


def share_of(distinct: pl.DataFrame, matches: pl.Expr) -> tuple[int, int]:
    """The rows whose value satisfies ``matches``, and all the rows."""
    total = int(distinct["count"].sum())
    found = distinct.filter(matches)["count"].sum()
    return int(found), total


def examples(
    distinct: pl.DataFrame, matches: pl.Expr, limit: int, width: int = 80
) -> list[str]:
    """Up to ``limit`` of the most frequent matching values, cut at ``width``.

    Equal counts are listed in sorted order, so a rerun lists the same values.
    """
    found = (
        distinct.filter(matches)
        .sort(["count", "value"], descending=[True, False])
        .head(limit)["value"]
        .to_list()
    )
    return [
        value if len(value) <= width else value[: width - 3] + "..." for value in found
    ]


# Words that stand for a missing value when they are written as text.
NULL_TOKENS = r"(?i)^\s*(|na|n/a|#n/a|nan|null|none|nil|missing|-|--|\?)\s*$"
_NUMBER = r"^\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*$"
_ZERO_PADDED = r"^\s*[+-]?0\d"
_BOOLEAN = r"(?i)^\s*(true|false|yes|no|t|f|y|n)\s*$"
# A cheap test that a value could be a date, so the formats are tried on few values.
_DATE_LIKE = (
    r"(?i)^\s*(?:\d{1,4}[-/. ]\d{1,2}[-/. ]\d{1,4}|\d{1,4}[-/. ]\d{1,2}[-/. ]"
    r"\d{1,4}[T ]\d{1,2}:\d{2}|\d{1,2}[- ](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|"
    r"nov|dec)[a-z]*[- ,]+\d{4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\.? \d{1,2},? \d{4})"
)
COMPACT_DATE = "%Y%m%d"
_FORMAT_SAMPLE = 2_000


def date_format_of(distinct: pl.DataFrame, formats: tuple[str, ...]) -> pl.DataFrame:
    """``distinct`` with a ``format`` column: the first of ``formats`` that parses.

    Null where none does. A value is tried only if it could be a date at all, and
    only with the formats that parse a seeded sample of at most 2 000 of those
    values, so a column of many distinct dates is parsed with the few formats it
    uses, not all of them. A format used by under one value in the sample's size
    can be missed; one that matters to a check (a twentieth of the values) is not.
    """
    value = pl.col("value").str.strip_chars()
    plausible = value.str.contains(_DATE_LIKE) | (
        value.str.contains(r"^\d{8}$") & pl.lit(COMPACT_DATE in formats)
    )
    candidates = distinct.filter(plausible)["value"].str.strip_chars()
    if candidates.len() > _FORMAT_SAMPLE:
        candidates = candidates.sample(_FORMAT_SAMPLE, seed=0)
    chosen = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.String))
    for fmt in formats:
        parsed = candidates.str.to_datetime(fmt, time_unit="us", strict=False)
        if parsed.null_count() == candidates.len():
            continue
        chosen = chosen.when(
            plausible
            & value.str.to_datetime(fmt, time_unit="us", strict=False).is_not_null()
        ).then(pl.lit(fmt))
    return distinct.with_columns(format=chosen.otherwise(None))


def value_kind(
    distinct: pl.DataFrame,
    formats: tuple[str, ...],
    compact_share: float = 0.05,
) -> pl.DataFrame:
    """``distinct`` with a ``kind`` and a ``format`` column: what each string reads as.

    The kind is one of ``null_token``, ``number``, ``boolean``, ``date`` or ``text``;
    ``format`` is the date format of a ``date``. A number is an integer or a decimal,
    with an optional exponent; a zero followed by a digit makes a code (a ZIP code, a
    zero-padded counter), which is ``text``, and so are the all-digit values of a
    length that some zero-padded value has. An eight-digit number that parses as
    ``%Y%m%d`` is a date only if the column holds other dates, at least
    ``compact_share`` of its values: alone, it is as likely a key.
    """
    dated = date_format_of(distinct, formats)
    value = pl.col("value")
    kind = (
        pl.when(value.str.contains(NULL_TOKENS))
        .then(pl.lit("null_token"))
        .when(value.str.contains(_NUMBER) & ~value.str.contains(_ZERO_PADDED))
        .then(pl.lit("number"))
        .when(value.str.contains(_BOOLEAN))
        .then(pl.lit("boolean"))
        .when(pl.col("format").is_not_null())
        .then(pl.lit("date"))
        .otherwise(pl.lit("text"))
    )
    kinded = dated.with_columns(kind=kind)
    # A fixed-width code with some zero-padded values is a code throughout.
    padded = kinded.filter(
        value.str.contains(_ZERO_PADDED) & value.str.contains(r"^\d+$")
    )["value"].str.len_chars()
    if padded.len():
        code = (
            (pl.col("kind") == "number")
            & value.str.contains(r"^\d+$")
            & value.str.len_chars().is_in(padded.unique().implode())
        )
        kinded = kinded.with_columns(
            kind=pl.when(code).then(pl.lit("text")).otherwise(pl.col("kind"))
        )
    total = int(kinded["count"].sum())
    dates = int(kinded.filter(pl.col("kind") == "date")["count"].sum())
    if not total or dates / total < compact_share:
        return kinded
    compact = (pl.col("kind") == "number") & (
        value.str.contains(r"^\d{8}$")
        & value.str.to_datetime(
            COMPACT_DATE, time_unit="us", strict=False
        ).is_not_null()
    )
    return kinded.with_columns(
        kind=pl.when(compact).then(pl.lit("date")).otherwise(pl.col("kind")),
        format=pl.when(compact).then(pl.lit(COMPACT_DATE)).otherwise(pl.col("format")),
    )
