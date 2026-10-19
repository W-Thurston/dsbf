# dsbf/profiling/values.py
"""Top values, string lengths, character classes and patterns of a column.

A column's distinct values and their counts are worked out once. Everything else
reads that small frame and weights each distinct value by its count, so the cost of
the string statistics follows the number of distinct values, not the number of rows.
"""

from dataclasses import dataclass

import polars as pl

from dsbf.model.profile import LengthStats, PatternCount, TextShape, ValueCount

DEFAULT_TOP_K = 10
MAX_TOP_VALUE_CHARS = 256

_UPPER_RUN = r"\p{Lu}+"
_LOWER_RUN = r"\p{Ll}+"
_DIGIT_RUN = r"\p{Nd}+"
_SYMBOL = r"[^\w\s]"
_EDGE_SPACE = r"^\s|\s$"


@dataclass(frozen=True, slots=True)
class StringStats:
    """What the distinct values of a string column say about its strings."""

    lengths: LengthStats | None
    avg_word_length: float | None
    shape: TextShape | None


def value_counts(series: pl.Series) -> pl.DataFrame:
    """The distinct non-null values and their counts, in order of first appearance.

    Columns are ``value`` and ``count``. Float NaN and infinities are left out, because
    a profile records the value of a top entry as a finite number.
    """
    values = series.drop_nulls()
    if values.dtype.is_float():
        values = values.filter(values.is_finite())
    elif values.dtype == pl.Decimal:
        values = values.cast(pl.String)
    return (
        values.to_frame("value")
        .group_by("value", maintain_order=True)
        .agg(pl.len().alias("count"))
    )


def top_values(
    counts: pl.DataFrame, k: int, *, min_count: int = 1
) -> tuple[ValueCount, ...]:
    """The ``k`` most frequent values; equal counts stay in order of first appearance.

    Only values seen at least ``min_count`` times are listed, and a string longer than
    ``MAX_TOP_VALUE_CHARS`` characters is never listed: it is not a label, and one
    such value would outweigh the rest of the profile.
    """
    if k <= 0:
        return ()
    keep = pl.col("count") >= min_count
    if counts.schema["value"] in (pl.String, pl.Categorical, pl.Enum):
        keep &= pl.col("value").cast(pl.String).str.len_chars() <= MAX_TOP_VALUE_CHARS
    ranked = (
        counts.filter(keep).sort("count", descending=True, maintain_order=True).head(k)
    )
    return tuple(
        ValueCount(value=value, count=count) for value, count in ranked.iter_rows()
    )


def string_stats(counts: pl.DataFrame, k: int) -> StringStats:
    """Lengths, word length and character shape of a column of strings.

    ``counts`` is ``value_counts`` of a String, Categorical or Enum column. Lengths
    cover every non-null string, empty ones included; the shape covers only non-empty
    strings, since an empty string has no characters to classify.
    """
    if counts.is_empty():
        return StringStats(None, None, None)
    value = pl.col("value").cast(pl.String)
    weight = pl.col("count").cast(pl.Int64)
    length = value.str.len_chars().cast(pl.Int64)
    non_empty = length > 0

    def weighted(flag: pl.Expr) -> pl.Expr:
        return (flag & non_empty).cast(pl.Int64).mul(weight).sum()

    totals = counts.select(
        min=length.min(),
        max=length.max(),
        chars=(length * weight).sum(),
        rows=weight.sum(),
        words=(value.str.count_matches(r"\S+").cast(pl.Int64) * weight).sum(),
        word_chars=(value.str.count_matches(r"\S").cast(pl.Int64) * weight).sum(),
        with_letters=weighted(value.str.contains(r"\p{L}")),
        with_digits=weighted(value.str.contains(r"\p{Nd}")),
        with_whitespace=weighted(value.str.contains(r"\s")),
        with_symbols=weighted(value.str.contains(_SYMBOL)),
        edge_whitespace=weighted(value.str.contains(_EDGE_SPACE)),
    ).row(0, named=True)

    lengths = LengthStats(
        min=totals["min"], max=totals["max"], mean=totals["chars"] / totals["rows"]
    )
    avg_word_length = (
        totals["word_chars"] / totals["words"] if totals["words"] else None
    )

    patterns = (
        counts.filter(non_empty)
        .select(pattern=_signature(value), count=weight)
        .group_by("pattern", maintain_order=True)
        .agg(pl.col("count").sum())
    )
    ranked = patterns.sort("count", descending=True, maintain_order=True).head(
        max(k, 0)
    )
    shape = TextShape(
        with_letters=totals["with_letters"],
        with_digits=totals["with_digits"],
        with_whitespace=totals["with_whitespace"],
        with_symbols=totals["with_symbols"],
        edge_whitespace=totals["edge_whitespace"],
        n_patterns=patterns.height,
        patterns=tuple(
            PatternCount(pattern=pattern, count=count)
            for pattern, count in ranked.iter_rows()
        ),
    )
    return StringStats(lengths, avg_word_length, shape)


def _signature(value: pl.Expr) -> pl.Expr:
    """``AB-123`` is ``A-9``: each run of one character class becomes one symbol."""
    return (
        value.str.replace_all(_UPPER_RUN, "A")
        .str.replace_all(_LOWER_RUN, "a")
        .str.replace_all(_DIGIT_RUN, "9")
    )
