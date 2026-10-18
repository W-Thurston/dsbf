"""Column and dataset profiles accept what the profiler produces and refuse the rest."""

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from dsbf.model import (
    BooleanProfile,
    CategoricalProfile,
    ColumnCounts,
    ColumnProfile,
    DatasetProfile,
    DatetimeProfile,
    Histogram,
    IdentifierProfile,
    LengthStats,
    Missingness,
    NullPair,
    NumericStats,
    PatternCount,
    Quantile,
    TextProfile,
    TextShape,
    UnknownProfile,
    ValueCount,
)

from .builders import (
    make_counts,
    make_dataset_profile,
    make_numeric,
    round_trip,
    strict_loads,
)

pytestmark = pytest.mark.unit

COLUMN = TypeAdapter(ColumnProfile)


def common(name="c", position=0, rows=10, nulls=0, **counts):
    return {
        "name": name,
        "position": position,
        "physical_dtype": "String",
        "counts": make_counts(rows=rows, nulls=nulls, **counts),
    }


# ── counts ───────────────────────────────────────────────────────────────────


def test_counts_cannot_exceed_the_rows():
    with pytest.raises(ValidationError, match="nulls is at most rows"):
        ColumnCounts(rows=5, nulls=6, distinct=0)
    with pytest.raises(ValidationError, match="distinct is at most"):
        ColumnCounts(rows=5, nulls=2, distinct=4)


def test_an_estimated_distinct_count_says_so():
    counts = ColumnCounts(rows=100, nulls=0, distinct=97, distinct_exact=False)

    assert round_trip(counts).distinct_exact is False
    assert ColumnCounts(rows=3, nulls=1, distinct=2).non_null == 2


# ── numeric ──────────────────────────────────────────────────────────────────


def test_numpy_statistics_are_accepted_and_serialise_as_numbers():
    values = np.array([1.0, 2.0, 4.0])
    stats = NumericStats(min=values.min(), max=values.max(), mean=values.mean())

    assert strict_loads(stats.model_dump_json())["mean"] == pytest.approx(7 / 3)


def test_undefined_statistics_become_null_and_are_named():
    stats = NumericStats(mean=1.0, std=math.nan, skewness=-math.inf)

    assert stats.std is None
    assert stats.skewness is None
    assert stats.mean == 1.0
    assert stats.non_finite_stats == ("skewness", "std")
    assert strict_loads(stats.model_dump_json())["std"] is None


def test_a_flagged_statistic_survives_a_round_trip():
    stats = NumericStats(std=math.nan)

    assert round_trip(stats) == stats


def test_a_quantile_must_be_finite():
    with pytest.raises(ValidationError):
        Quantile(p=0.5, value=math.nan)
    with pytest.raises(ValidationError):
        Quantile(p=1.5, value=1.0)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"min": 3.0, "max": 2.0}, "min is at most max"),
        (
            {
                "quantiles": (Quantile(p=0.5, value=1.0), Quantile(p=0.5, value=2.0)),
                "quantile_method": "exact",
            },
            "increasing order of p",
        ),
        (
            {
                "quantiles": (Quantile(p=0.1, value=5.0), Quantile(p=0.9, value=4.0)),
                "quantile_method": "exact",
            },
            "do not decrease",
        ),
        ({"quantiles": (Quantile(p=0.5, value=1.0),)}, "how they were computed"),
    ],
)
def test_statistics_are_ordered(fields, message):
    with pytest.raises(ValidationError, match=message):
        NumericStats(**fields)


def test_a_histogram_records_its_rule_and_equal_bins():
    histogram = Histogram(edges=(0.0, 0.1, 0.2, 0.3), counts=(1, 2, 3), rule="fd")

    assert histogram.total == 6
    assert round_trip(histogram) == histogram


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"edges": (0.0, 1.0), "counts": (1, 2)}, "one more edge"),
        ({"edges": (0.0,), "counts": ()}, "at least one bin"),
        ({"edges": (0.0, 1.0, 1.0), "counts": (1, 1)}, "increase strictly"),
        ({"edges": (0.0, 1.0, 2.0), "counts": (1, -1)}, "not negative"),
        ({"edges": (0.0, 1.0, 5.0), "counts": (1, 1)}, "equal width"),
    ],
)
def test_a_histogram_must_be_well_formed(fields, message):
    with pytest.raises(ValidationError, match=message):
        Histogram(rule="fd", **fields)


def test_a_histogram_needs_a_rule():
    with pytest.raises(ValidationError):
        Histogram.model_validate({"edges": (0.0, 1.0), "counts": (1,)})
    with pytest.raises(ValidationError):
        Histogram.model_validate(
            {"edges": (0.0, 1.0), "counts": (1,), "rule": "percentiles"}
        )


def test_numeric_counts_cannot_exceed_the_non_null_rows():
    with pytest.raises(ValidationError, match="zeros and non_finite"):
        make_numeric(zeros=10)
    with pytest.raises(ValidationError, match="histogram counts"):
        too_many = Histogram(edges=(0.0, 1.0), counts=(10,), rule="sturges")
        make_numeric(histogram=too_many)


def test_a_numeric_column_with_no_finite_value_has_no_statistics():
    column = make_numeric(stats=None, histogram=None, non_finite=8)

    assert round_trip(column) == column


# ── the other variants ───────────────────────────────────────────────────────


def test_top_values_are_ordered_and_bounded():
    top = (ValueCount(value="a", count=5), ValueCount(value="b", count=3))
    column = CategoricalProfile(**common(distinct=4), top_values=top)
    assert round_trip(column).top_values == top

    with pytest.raises(ValidationError, match="decreasing count"):
        CategoricalProfile(**common(), top_values=top[::-1])
    with pytest.raises(ValidationError, match="at most the non-null rows"):
        CategoricalProfile(**common(rows=7, distinct=4), top_values=top)
    with pytest.raises(ValidationError, match="no more top values"):
        CategoricalProfile(**common(distinct=1), top_values=top)


def test_top_values_keep_their_type():
    top = (
        ValueCount(value=True, count=4),
        ValueCount(value=3, count=3),
        ValueCount(value=1.5, count=2),
        ValueCount(value="", count=1),
    )

    restored = round_trip(CategoricalProfile(**common(distinct=4), top_values=top))

    assert [type(v.value) for v in restored.top_values] == [bool, int, float, str]
    with pytest.raises(ValidationError):
        ValueCount.model_validate({"value": None, "count": 1})
    with pytest.raises(ValidationError):
        ValueCount.model_validate({"value": "a", "count": 0})


def test_length_statistics_are_ordered():
    LengthStats(min=1, max=9, mean=4.5)
    with pytest.raises(ValidationError, match="min is at most max"):
        LengthStats(min=5, max=1, mean=3.0)
    with pytest.raises(ValidationError, match="between min and max"):
        LengthStats(min=1, max=2, mean=5.0)


def test_a_boolean_column_splits_the_non_null_rows():
    column = BooleanProfile(**common(rows=10, nulls=2), true_count=5, false_count=3)
    assert round_trip(column) == column

    with pytest.raises(ValidationError, match="add up to the non-null rows"):
        BooleanProfile(**common(rows=10, nulls=2), true_count=5, false_count=4)


def test_a_datetime_column_has_an_ordered_range():
    column = DatetimeProfile(
        **common(),
        min=datetime(2020, 1, 1, tzinfo=UTC),
        max=datetime(2020, 12, 31, tzinfo=UTC),
        timezone="UTC",
        inferred_frequency="D",
        n_gaps=0,
        largest_gap_seconds=86400.0,
    )
    assert round_trip(column) == column

    with pytest.raises(ValidationError, match="min is at most max"):
        DatetimeProfile(**common(), min=datetime(2021, 1, 1), max=datetime(2020, 1, 1))
    with pytest.raises(ValidationError, match="naive or both aware"):
        DatetimeProfile(
            **common(), min=datetime(2020, 1, 1), max=datetime(2021, 1, 1, tzinfo=UTC)
        )


def test_text_identifier_and_unknown_columns_round_trip():
    for column in (
        TextProfile(
            **common(),
            lengths=LengthStats(min=2, max=40, mean=11.5),
            avg_word_length=4.2,
            empty_strings=1,
        ),
        IdentifierProfile(**common()),
        UnknownProfile(**common()),
    ):
        assert round_trip(column) == column


def test_empty_strings_cannot_exceed_the_non_null_rows():
    with pytest.raises(ValidationError, match="empty_strings"):
        TextProfile(**common(rows=4, nulls=2, distinct=1), empty_strings=3)


def shape(**fields) -> TextShape:
    base: dict[str, Any] = {
        "with_letters": 3,
        "with_digits": 1,
        "with_whitespace": 0,
        "with_symbols": 0,
        "edge_whitespace": 0,
        "n_patterns": 2,
        "patterns": (
            PatternCount(pattern="Aa", count=2),
            PatternCount(pattern="9", count=1),
        ),
    }
    return TextShape.model_validate({**base, **fields})


def test_a_text_shape_round_trips_on_both_string_profiles():
    for column in (
        CategoricalProfile(**common(rows=4, distinct=3), shape=shape()),
        TextProfile(**common(rows=4, distinct=3), shape=shape()),
    ):
        assert round_trip(column) == column


def test_a_shape_is_optional():
    assert CategoricalProfile(**common()).shape is None
    assert TextProfile(**common()).shape is None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"with_letters": 5}, "with_letters is at most"),
        ({"with_digits": 5}, "with_digits is at most"),
        ({"with_whitespace": 5}, "with_whitespace is at most"),
        ({"with_symbols": 5}, "with_symbols is at most"),
        ({"edge_whitespace": 5}, "edge_whitespace is at most"),
        ({"n_patterns": 5}, "n_patterns is at most"),
        (
            {"patterns": (PatternCount(pattern="a", count=4),)},
            "pattern counts add up",
        ),
    ],
)
def test_shape_counts_cannot_exceed_the_non_empty_values(fields, message):
    with pytest.raises(ValidationError, match=message):
        # Four rows, one empty string: three values have characters.
        CategoricalProfile(
            **common(rows=4, distinct=3), empty_strings=1, shape=shape(**fields)
        )


def test_patterns_are_listed_by_decreasing_count_and_within_n_patterns():
    with pytest.raises(ValidationError, match="decreasing count"):
        shape(
            patterns=(
                PatternCount(pattern="a", count=1),
                PatternCount(pattern="9", count=2),
            )
        )
    with pytest.raises(ValidationError, match="n_patterns"):
        shape(n_patterns=1)


def test_a_pattern_needs_at_least_one_value():
    with pytest.raises(ValidationError):
        PatternCount.model_validate({"pattern": "a", "count": 0})


def test_a_semantic_type_is_chosen_by_its_tag():
    for column in (make_numeric(), IdentifierProfile(**common())):
        restored = COLUMN.validate_json(column.model_dump_json())

        assert type(restored) is type(column)
        assert restored.semantic_type == column.semantic_type

    with pytest.raises(ValidationError):
        COLUMN.validate_python({**common(), "semantic_type": "ordinal"})


def test_an_unknown_field_is_refused():
    with pytest.raises(ValidationError):
        IdentifierProfile.model_validate({**common(), "mean": 1.0})


# ── dataset ──────────────────────────────────────────────────────────────────


def test_a_dataset_profile_looks_columns_up_by_name():
    profile = make_dataset_profile(
        [make_numeric("a", 0), make_numeric("b", 1)],
        duplicate_rows=1,
        rows_with_any_null=4,
        approx_memory_bytes=2048,
    )

    assert profile.n_columns == 2
    assert profile.column("b").position == 1
    with pytest.raises(KeyError):
        profile.column("c")
    assert round_trip(profile) == profile


@pytest.mark.parametrize("name", ["a|b", "[x]", "naïve — 名前", "", "a b", "'quoted'"])
def test_awkward_column_names_stay_intact(name):
    profile = make_dataset_profile([make_numeric(name)])

    restored = round_trip(profile)

    assert restored.column(name).name == name


def test_a_dataset_profile_with_no_columns_is_valid():
    assert round_trip(DatasetProfile(n_rows=0, columns=())).n_columns == 0


@pytest.mark.parametrize(
    ("columns", "fields", "message"),
    [
        ([make_numeric("a"), make_numeric("a", 1)], {}, "distinct"),
        ([make_numeric("a", 1)], {}, "position"),
        ([make_numeric("a", rows=9)], {}, "counts 9 rows"),
        ([make_numeric("a")], {"duplicate_rows": 11}, "duplicate_rows"),
        ([make_numeric("a")], {"rows_with_any_null": 11}, "rows_with_any_null"),
    ],
)
def test_a_dataset_profile_must_agree_with_its_columns(columns, fields, message):
    with pytest.raises(ValidationError, match=message):
        make_dataset_profile(columns, **fields)


# ── missingness ──────────────────────────────────────────────────────────────


def _missingness(*pairs, with_nulls=2, considered=2):
    return Missingness(
        columns_with_nulls=with_nulls, columns_considered=considered, pairs=pairs
    )


def test_missingness_round_trips_pairs_of_awkward_names():
    pair = NullPair(a="a|b", b="[x]", both_null=1, jaccard=1.0)
    profile = make_dataset_profile(
        [make_numeric("a|b"), make_numeric("[x]", 1)],
        missingness=_missingness(pair),
    )

    assert round_trip(profile).missingness.pairs == (pair,)


@pytest.mark.parametrize(
    ("pair", "message"),
    [
        (NullPair(a="a", b="a", both_null=1, jaccard=1.0), "twice"),
        (NullPair(a="a", b="z", both_null=1, jaccard=1.0), "unknown column"),
        (NullPair(a="a", b="b", both_null=2, jaccard=1.0), "more null rows"),
        (NullPair(a="b", b="a", both_null=1, jaccard=1.0), "column order"),
    ],
)
def test_a_null_pair_must_agree_with_the_columns(pair, message):
    columns = [make_numeric("a"), make_numeric("b", 1)]

    with pytest.raises(ValidationError, match=message):
        make_dataset_profile(columns, missingness=_missingness(pair))


def test_missingness_counts_must_agree_with_the_columns():
    columns = [make_numeric("a"), make_numeric("b", 1)]

    with pytest.raises(ValidationError, match="columns_with_nulls"):
        make_dataset_profile(columns, missingness=_missingness(with_nulls=1))
    with pytest.raises(ValidationError, match="columns_considered"):
        _missingness(with_nulls=1, considered=2)


@pytest.mark.parametrize("jaccard", [-0.1, 1.1, float("nan")])
def test_a_jaccard_outside_zero_to_one_is_refused(jaccard):
    with pytest.raises(ValidationError):
        NullPair(a="a", b="b", both_null=1, jaccard=jaccard)
