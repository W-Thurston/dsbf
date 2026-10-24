# dsbf/model/profile.py
"""What the profiling pass learns about each column and about the dataset."""

import math
import re
from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from dsbf.model.base import Contract, FiniteFloat
from dsbf.model.scrub import json_safe
from dsbf.model.vocabulary import Confidence

SemanticType = Literal[
    "numeric", "categorical", "boolean", "datetime", "text", "identifier", "unknown"
]

# Every statistic below is computed over the column's non-null values.
_NUMERIC_STATS = ("min", "max", "mean", "std", "skewness", "kurtosis")


class ColumnCounts(Contract):
    """How many rows a column has, how many are null, and how many values differ."""

    rows: int = Field(ge=0, description="Rows profiled, after any sampling.")
    nulls: int = Field(ge=0)
    distinct: int = Field(ge=0, description="Distinct non-null values.")
    distinct_exact: bool = Field(
        default=True, description="False when `distinct` is an estimate."
    )

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if self.nulls > self.rows:
            raise ValueError("nulls is at most rows")
        if self.distinct > self.non_null:
            raise ValueError("distinct is at most the number of non-null rows")
        return self

    @property
    def non_null(self) -> int:
        return self.rows - self.nulls


class Quantile(Contract):
    """The value below which a share ``p`` of the non-null values fall."""

    p: FiniteFloat = Field(ge=0, le=1)
    value: FiniteFloat


class NumericStats(Contract):
    """Summary statistics of a numeric column's finite, non-null values."""

    min: FiniteFloat | None = None
    max: FiniteFloat | None = None
    mean: FiniteFloat | None = None
    std: FiniteFloat | None = None
    skewness: FiniteFloat | None = None
    kurtosis: FiniteFloat | None = None
    non_finite_stats: tuple[str, ...] = Field(
        default=(),
        description="Statistics that were NaN or infinite, so they are null here.",
    )
    quantiles: tuple[Quantile, ...] = Field(
        default=(), description="In increasing order of `p`."
    )
    quantile_method: Literal["exact", "tdigest"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _scrub_stats(cls, data: Any) -> Any:
        """Null undefined statistics and name them (never raising)."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        undefined = set(data.get("non_finite_stats") or ())
        for name in _NUMERIC_STATS:
            if name in data:
                data[name], is_undefined = json_safe(data[name])
                if is_undefined:
                    undefined.add(name)
        data["non_finite_stats"] = tuple(sorted(undefined))
        return data

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min is at most max")
        points = [q.p for q in self.quantiles]
        if any(a >= b for a, b in zip(points, points[1:], strict=False)):
            raise ValueError("quantiles are listed in increasing order of p")
        values = [q.value for q in self.quantiles]
        if any(a > b for a, b in zip(values, values[1:], strict=False)):
            raise ValueError("quantile values do not decrease as p grows")
        if self.quantiles and self.quantile_method is None:
            raise ValueError("quantiles record how they were computed")
        return self


class Histogram(Contract):
    """Equal-width bins over a numeric column, with the rule that chose them."""

    edges: tuple[FiniteFloat, ...] = Field(description="One more edge than bins.")
    counts: tuple[int, ...] = Field(description="Values per bin; the last is closed.")
    rule: Literal["fd", "sturges", "scott", "sqrt", "fixed"] = Field(
        description=(
            "The rule that proposed the number of bins; a count over the profile's "
            "bin limit is clipped to it."
        )
    )

    @model_validator(mode="after")
    def _check_bins(self) -> Self:
        if len(self.edges) != len(self.counts) + 1:
            raise ValueError("a histogram has one more edge than it has bins")
        if not self.counts:
            raise ValueError("a histogram has at least one bin")
        if any(a >= b for a, b in zip(self.edges, self.edges[1:], strict=False)):
            raise ValueError("histogram edges increase strictly")
        if any(count < 0 for count in self.counts):
            raise ValueError("histogram counts are not negative")
        widths = [b - a for a, b in zip(self.edges, self.edges[1:], strict=False)]
        scale = max(abs(edge) for edge in self.edges)
        if max(widths) - min(widths) > 1e-6 * scale:
            raise ValueError("histogram bins have equal width")
        return self

    @property
    def total(self) -> int:
        return sum(self.counts)


class ValueCount(Contract):
    """One value and how many rows hold it."""

    value: bool | int | FiniteFloat | str
    count: int = Field(ge=1)


class LengthStats(Contract):
    """Lengths, in characters, of a column's non-null strings."""

    min: int = Field(ge=0)
    max: int = Field(ge=0)
    mean: FiniteFloat = Field(ge=0)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.min > self.max:
            raise ValueError("min is at most max")
        if not self.min <= self.mean <= self.max:
            raise ValueError("mean lies between min and max")
        return self


class PatternCount(Contract):
    """A pattern signature and how many values have it."""

    pattern: str = Field(
        description=(
            "Each run of uppercase letters is `A`, of lowercase letters `a` and of "
            "digits `9`; other characters stand for themselves."
        )
    )
    count: int = Field(ge=1)


class TextShape(Contract):
    """What the characters of a string column's non-empty values look like.

    Every count is of non-null, non-empty values, so none exceeds the number of those.
    """

    with_letters: int = Field(ge=0, description="Values with at least one letter.")
    with_digits: int = Field(ge=0, description="Values with at least one digit.")
    with_whitespace: int = Field(ge=0, description="Values with a space or tab.")
    with_symbols: int = Field(
        ge=0, description="Values with a character that is not a word or space."
    )
    edge_whitespace: int = Field(
        ge=0, description="Values that start or end with whitespace."
    )
    n_patterns: int = Field(ge=0, description="Distinct pattern signatures.")
    patterns: tuple[PatternCount, ...] = Field(
        default=(), description="The most common signatures, by decreasing count."
    )

    @model_validator(mode="after")
    def _check_patterns(self) -> Self:
        if len(self.patterns) > self.n_patterns:
            raise ValueError("there are no more patterns listed than n_patterns")
        ordered = [p.count for p in self.patterns]
        if any(a < b for a, b in zip(ordered, ordered[1:], strict=False)):
            raise ValueError("patterns are listed by decreasing count")
        return self

    def check_against(self, counts: ColumnCounts, empty_strings: int) -> None:
        """Raise if the counts could not come from a column with these counts."""
        candidates = counts.non_null - empty_strings
        for name in (
            "with_letters",
            "with_digits",
            "with_whitespace",
            "with_symbols",
            "edge_whitespace",
        ):
            if getattr(self, name) > candidates:
                raise ValueError(f"{name} is at most the non-empty values")
        if self.n_patterns > candidates:
            raise ValueError("n_patterns is at most the non-empty values")
        if sum(p.count for p in self.patterns) > candidates:
            raise ValueError("pattern counts add up to at most the non-empty values")


def _check_top_values(top_values: tuple[ValueCount, ...], counts: ColumnCounts) -> None:
    if len(top_values) > counts.distinct:
        raise ValueError("there are no more top values than distinct values")
    ordered = [v.count for v in top_values]
    if any(a < b for a, b in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("top values are listed by decreasing count")
    if sum(ordered) > counts.non_null:
        raise ValueError("top value counts add up to at most the non-null rows")


_CONTENT_HASH = re.compile(r"[0-9a-f]{16}")
# Entropy is a sum of floating-point terms, so it may exceed log2(distinct) by rounding.
_ENTROPY_SLACK = 1e-9


def _check_entropy(entropy_bits: float | None, counts: ColumnCounts) -> None:
    if entropy_bits is None:
        return
    if not counts.non_null:
        raise ValueError("a column with no values has no entropy")
    if entropy_bits > math.log2(max(counts.distinct, 1)) + _ENTROPY_SLACK:
        raise ValueError("entropy_bits is at most log2 of the distinct values")


class _ColumnBase(Contract):
    """What every column profile carries, whatever the column holds."""

    name: str
    position: int = Field(ge=0, description="Zero-based place in the dataset.")
    physical_dtype: str = Field(description="The storage type, as a string.")
    semantic_confidence: Confidence = Field(
        default=Confidence.HIGH, description="How firmly the semantic type is held."
    )
    semantic_sample_rows: int | None = Field(
        default=None, ge=0, description="Rows the semantic type was inferred from."
    )
    counts: ColumnCounts
    content_hash: str | None = Field(
        default=None,
        description=(
            "A 64-bit digest of the column's values in row order, as 16 hex digits; "
            "None when the values cannot be hashed. Equal hashes mark candidate "
            "duplicates only: they also need equal `physical_dtype` and a "
            "comparison of the values, because the digest is 64 bits and ignores "
            "the dtype."
        ),
    )
    name_encoding_damage: bool = Field(
        default=False,
        description="The name holds U+FFFD (a character that failed to decode) or NUL.",
    )
    values_with_replacement_char: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Values holding U+FFFD, the mark a lossy decode leaves; None unless the "
            "column holds strings."
        ),
    )
    values_with_nul: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Values holding a NUL character, as a UTF-16 file read as UTF-8 leaves; "
            "None unless the column holds strings."
        ),
    )

    @model_validator(mode="after")
    def _check_common(self) -> Self:
        if self.content_hash is not None and not _CONTENT_HASH.fullmatch(
            self.content_hash
        ):
            raise ValueError("content_hash is 16 lowercase hex digits")
        for name in ("values_with_replacement_char", "values_with_nul"):
            value = getattr(self, name)
            if value is not None and value > self.counts.non_null:
                raise ValueError(f"{name} is at most the non-null rows")
        return self


class NumericProfile(_ColumnBase):
    semantic_type: Literal["numeric"] = "numeric"
    zeros: int = Field(ge=0)
    non_finite: int = Field(
        default=0,
        ge=0,
        description="NaN and infinite values, counted apart from nulls.",
    )
    stats: NumericStats | None = Field(
        default=None, description="None when there is no finite value to summarise."
    )
    histogram: Histogram | None = None

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        non_null = self.counts.non_null
        if self.zeros > non_null or self.non_finite > non_null:
            raise ValueError("zeros and non_finite are at most the non-null rows")
        if self.histogram and self.histogram.total > non_null - self.non_finite:
            raise ValueError("histogram counts add up to at most the finite values")
        return self


class CategoricalProfile(_ColumnBase):
    semantic_type: Literal["categorical"] = "categorical"
    top_values: tuple[ValueCount, ...] = ()
    mode_count: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Rows holding the most frequent non-null value, whether or not it is "
            "listed in `top_values`: a string over 256 characters never is. None when "
            "the column has no value."
        ),
    )
    lengths: LengthStats | None = None
    empty_strings: int = Field(default=0, ge=0)
    shape: TextShape | None = Field(
        default=None, description="Present for string columns only."
    )
    entropy_bits: FiniteFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "Shannon entropy, in bits, of the whole distribution of non-null values "
            "(not only the top values); None when there are none."
        ),
    )

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        _check_entropy(self.entropy_bits, self.counts)
        _check_top_values(self.top_values, self.counts)
        if self.mode_count is not None:
            if self.mode_count > self.counts.non_null:
                raise ValueError("mode_count is at most the non-null rows")
            if self.top_values and self.top_values[0].count > self.mode_count:
                raise ValueError("mode_count is at least the count of the top value")
        if self.empty_strings > self.counts.non_null:
            raise ValueError("empty_strings is at most the non-null rows")
        if self.shape is not None:
            self.shape.check_against(self.counts, self.empty_strings)
        return self


class BooleanProfile(_ColumnBase):
    semantic_type: Literal["boolean"] = "boolean"
    true_count: int = Field(ge=0)
    false_count: int = Field(ge=0)
    entropy_bits: FiniteFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "Shannon entropy, in bits, of the whole distribution of non-null values "
            "(not only the top values); None when there are none."
        ),
    )

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        _check_entropy(self.entropy_bits, self.counts)
        if self.true_count + self.false_count != self.counts.non_null:
            raise ValueError("true_count and false_count add up to the non-null rows")
        return self


class DatetimeProfile(_ColumnBase):
    semantic_type: Literal["datetime"] = "datetime"
    min: datetime | None = None
    max: datetime | None = None
    timezone: str | None = None
    inferred_frequency: str | None = Field(
        default=None,
        description="An offset alias such as `D`, when the spacing is regular.",
    )
    n_gaps: int | None = Field(default=None, ge=0)
    largest_gap_seconds: FiniteFloat | None = Field(default=None, ge=0)
    format: str | None = Field(
        default=None,
        description="The strptime format the text values were read with; None for a "
        "date or datetime column, and for Unix timestamps.",
    )
    epoch_unit: Literal["s", "ms"] | None = Field(
        default=None, description="The unit of Unix timestamps; None otherwise."
    )
    unparsed: int = Field(
        default=0,
        ge=0,
        description="Non-null values that did not read as an instant: text outside "
        "`format`, or a timestamp out of range. 0 for a date or datetime column.",
    )
    time_of_day: bool | None = Field(
        default=None,
        description="Some instant is not at midnight (local time of a time zone); "
        "None when there are no instants.",
    )
    distinct_years: int | None = Field(default=None, ge=1)
    distinct_months: int | None = Field(
        default=None, ge=1, le=12, description="Distinct months of the year."
    )
    distinct_weekdays: int | None = Field(
        default=None, ge=1, le=7, description="Distinct days of the week."
    )
    distinct_hours: int | None = Field(
        default=None, ge=1, le=24, description="Distinct hours of the day."
    )

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.unparsed > self.counts.non_null:
            raise ValueError("unparsed is at most the non-null rows")
        if self.min is not None and self.max is not None:
            if (self.min.tzinfo is None) != (self.max.tzinfo is None):
                raise ValueError("min and max are both naive or both aware")
            if self.min > self.max:
                raise ValueError("min is at most max")
        return self


class TextProfile(_ColumnBase):
    semantic_type: Literal["text"] = "text"
    top_values: tuple[ValueCount, ...] = ()
    lengths: LengthStats | None = None
    avg_word_length: FiniteFloat | None = Field(
        default=None,
        ge=0,
        description="Mean characters per whitespace-delimited word.",
    )
    empty_strings: int = Field(default=0, ge=0)
    shape: TextShape | None = None
    entropy_bits: FiniteFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "Shannon entropy, in bits, of the whole distribution of non-null values "
            "(not only the top values); None when there are none."
        ),
    )

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        _check_entropy(self.entropy_bits, self.counts)
        _check_top_values(self.top_values, self.counts)
        if self.empty_strings > self.counts.non_null:
            raise ValueError("empty_strings is at most the non-null rows")
        if self.shape is not None:
            self.shape.check_against(self.counts, self.empty_strings)
        return self


class IdentifierProfile(_ColumnBase):
    """A column that names rows (for example a key); only the common fields apply."""

    semantic_type: Literal["identifier"] = "identifier"


class UnknownProfile(_ColumnBase):
    """A column whose meaning could not be inferred."""

    semantic_type: Literal["unknown"] = "unknown"


ColumnProfile = Annotated[
    NumericProfile
    | CategoricalProfile
    | BooleanProfile
    | DatetimeProfile
    | TextProfile
    | IdentifierProfile
    | UnknownProfile,
    Field(discriminator="semantic_type"),
]


class NullPair(Contract):
    """Two columns that are null in the same rows."""

    a: str = Field(description="The earlier column in the dataset's column order.")
    b: str
    both_null: int = Field(ge=1, description="Rows where both columns are null.")
    jaccard: FiniteFloat = Field(
        ge=0,
        le=1,
        description="Rows null in both over rows null in either: 1 means the "
        "columns are null in exactly the same rows.",
    )


class Missingness(Contract):
    """Which columns go missing together."""

    columns_with_nulls: int = Field(ge=0)
    columns_considered: int = Field(
        ge=0,
        description="Columns paired: the ones with the most nulls, when there are "
        "more than the pairing limit.",
    )
    pairs: tuple[NullPair, ...] = Field(
        description="The strongest co-occurring pairs, most similar first; only "
        "pairs that share at least one null row."
    )

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if self.columns_considered > self.columns_with_nulls:
            raise ValueError("columns_considered is at most columns_with_nulls")
        return self


class DatasetProfile(Contract):
    """The profile of a whole dataset: its shape and one profile per column."""

    n_rows: int = Field(ge=0, description="Rows profiled, after any sampling.")
    approx_memory_bytes: int | None = Field(default=None, ge=0)
    duplicate_rows: int | None = Field(
        default=None,
        ge=0,
        description="Rows equal to an earlier row; None if not computed.",
    )
    rows_with_any_null: int | None = Field(default=None, ge=0)
    missingness: Missingness | None = Field(
        default=None, description="Null co-occurrence; None if not computed."
    )
    columns: tuple[ColumnProfile, ...]

    @model_validator(mode="after")
    def _check_columns(self) -> Self:
        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("column names in a profile are distinct")
        for index, column in enumerate(self.columns):
            if column.position != index:
                raise ValueError(
                    f"column {column.name!r} is at position {column.position} "
                    f"but listed at {index}"
                )
            if column.counts.rows != self.n_rows:
                raise ValueError(
                    f"column {column.name!r} counts {column.counts.rows} rows, "
                    f"the profile {self.n_rows}"
                )
        for name in ("duplicate_rows", "rows_with_any_null"):
            value = getattr(self, name)
            if value is not None and value > self.n_rows:
                raise ValueError(f"{name} is at most n_rows")
        if self.missingness is not None:
            self._check_missingness(self.missingness)
        return self

    def _check_missingness(self, missingness: Missingness) -> None:
        by_name = {column.name: column for column in self.columns}
        if missingness.columns_with_nulls != sum(
            column.counts.nulls > 0 for column in self.columns
        ):
            raise ValueError("columns_with_nulls disagrees with the column counts")
        for pair in missingness.pairs:
            if pair.a == pair.b:
                raise ValueError(f"a null pair names {pair.a!r} twice")
            for name in (pair.a, pair.b):
                if name not in by_name:
                    raise ValueError(f"a null pair names unknown column {name!r}")
                if pair.both_null > by_name[name].counts.nulls:
                    raise ValueError(
                        f"pair {pair.a!r}, {pair.b!r} shares more null rows than "
                        f"{name!r} has nulls"
                    )
            if by_name[pair.a].position > by_name[pair.b].position:
                raise ValueError(f"pair {pair.a!r}, {pair.b!r} is out of column order")

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def column(self, name: str) -> ColumnProfile:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)
