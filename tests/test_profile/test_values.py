"""Top values, lengths, character classes and patterns of string columns."""

from decimal import Decimal

import polars as pl
import pytest

from dsbf.data import load
from dsbf.model.profile import CategoricalProfile, TextProfile
from dsbf.model.vocabulary import Confidence
from dsbf.profile import ColumnType, profile_columns
from dsbf.profile.values import (
    MAX_TOP_VALUE_CHARS,
    string_stats,
    top_values,
    value_counts,
)

pytestmark = pytest.mark.unit


def _top(values, k=10, dtype=None, **kwargs):
    counts = value_counts(pl.Series(values, dtype=dtype))
    return [(v.value, v.count) for v in top_values(counts, k, **kwargs)]


def _stats(values, k=10, dtype=None):
    return string_stats(value_counts(pl.Series(values, dtype=dtype)), k)


def _categorical(values, dtype=None, **kwargs) -> CategoricalProfile:
    frame = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
    (profile,) = profile_columns(load(frame), **kwargs)
    assert isinstance(profile, CategoricalProfile)
    return profile


class TestTopValues:
    def test_most_frequent_first_with_ties_in_order_of_first_appearance(self):
        assert _top(["b", "a", "c", "a", "c", "d"]) == [
            ("a", 2),
            ("c", 2),
            ("b", 1),
            ("d", 1),
        ]

    def test_nulls_are_not_values(self):
        assert _top(["a", None, "a", None, None]) == [("a", 2)]

    def test_only_k_values_are_kept(self):
        assert _top(list("aaabbc"), k=2) == [("a", 3), ("b", 2)]
        assert _top(list("aaabbc"), k=0) == []

    def test_a_minimum_count_drops_one_off_values(self):
        assert _top(list("aabcd"), min_count=2) == [("a", 2)]

    def test_a_string_over_the_length_limit_is_not_listed(self):
        long = "x" * (MAX_TOP_VALUE_CHARS + 1)
        exact = "y" * MAX_TOP_VALUE_CHARS
        assert _top([long, long, exact, "a"]) == [(exact, 1), ("a", 1)]

    def test_values_keep_their_type(self):
        assert _top([1, 1, 2], dtype=pl.Int64) == [(1, 2), (2, 1)]
        assert _top([1.5, 1.5, 2.5], dtype=pl.Float64) == [(1.5, 2), (2.5, 1)]
        assert _top([True, True, False]) == [(True, 2), (False, 1)]

    def test_non_finite_floats_are_left_out(self):
        values = [float("nan"), float("nan"), float("inf"), 1.0]
        assert _top(values, dtype=pl.Float64) == [(1.0, 1)]

    def test_decimals_are_listed_as_text(self):
        values = [Decimal("1.50"), Decimal("1.50"), Decimal("2.25")]
        assert _top(values, dtype=pl.Decimal(10, 2)) == [("1.50", 2), ("2.25", 1)]

    def test_categorical_and_enum_columns_list_their_labels(self):
        assert _top(["x", "y", "x"], dtype=pl.Categorical) == [("x", 2), ("y", 1)]
        assert _top(["x", "y", "x"], dtype=pl.Enum(["x", "y"])) == [("x", 2), ("y", 1)]


class TestInProfile:
    def test_a_categorical_column_lists_its_top_values(self):
        profile = _categorical(["a", "b", "a", None, "c", "a", "b"])
        assert [(v.value, v.count) for v in profile.top_values] == [
            ("a", 3),
            ("b", 2),
            ("c", 1),
        ]

    def test_integer_codes_list_their_top_values_without_string_statistics(self):
        profile = _categorical([0, 1, 1, 0, 1, 0, 1])
        assert [(v.value, v.count) for v in profile.top_values] == [(1, 4), (0, 3)]
        assert profile.lengths is None
        assert profile.shape is None

    def test_top_k_zero_lists_nothing(self):
        assert _categorical(["a", "b", "a"], top_k=0).top_values == ()

    def test_a_negative_top_k_is_refused(self):
        with pytest.raises(ValueError, match="top_k"):
            _categorical(["a", "b"], top_k=-1)

    def test_a_text_column_lists_only_values_that_repeat(self):
        repeated = [f"a sentence long enough to be text, number {i}" for i in range(30)]
        one_offs = [f"a one-off sentence long enough to be text, {i}" for i in range(5)]
        frame = pl.DataFrame({"x": [*repeated] * 3 + one_offs})
        (profile,) = profile_columns(load(frame), top_k=100)
        assert isinstance(profile, TextProfile)
        assert [v.value for v in profile.top_values] == repeated
        assert {v.count for v in profile.top_values} == {3}

    def test_a_text_column_with_no_repeats_has_no_top_values(self):
        frame = pl.DataFrame({"x": [f"free text value {i} " * 5 for i in range(40)]})
        forced = {"x": ColumnType("text", Confidence.HIGH, "String", "forced")}
        (profile,) = profile_columns(load(frame), forced)
        assert isinstance(profile, TextProfile)
        assert profile.top_values == ()
        assert profile.avg_word_length is not None


class TestLengths:
    def test_lengths_are_characters_and_cover_every_non_null_string(self):
        stats = _stats(["ab", "abcd", "", "é", None, "ab"])
        assert stats.lengths is not None
        assert (stats.lengths.min, stats.lengths.max) == (0, 4)
        assert stats.lengths.mean == pytest.approx((2 + 4 + 0 + 1 + 2) / 5)

    def test_an_empty_column_has_no_statistics(self):
        stats = _stats([None, None], dtype=pl.String)
        assert (stats.lengths, stats.avg_word_length, stats.shape) == (None, None, None)

    def test_the_average_word_length_counts_only_non_space_characters(self):
        stats = _stats(["aa bbbb", "c"])
        assert stats.avg_word_length == pytest.approx((2 + 4 + 1) / 3)

    def test_a_column_of_blanks_has_no_words(self):
        assert _stats(["  ", " "]).avg_word_length is None


class TestShape:
    def test_character_classes_count_values_not_characters(self):
        shape = _stats(["abc", "a1", "12", "a b", "a-b", " pad", "pad "]).shape
        assert shape is not None
        assert shape.with_letters == 6
        assert shape.with_digits == 2
        assert shape.with_whitespace == 3
        assert shape.with_symbols == 1
        assert shape.edge_whitespace == 2

    def test_empty_strings_are_left_out_of_the_shape(self):
        shape = _stats(["", "", "ab"]).shape
        assert shape is not None
        assert (shape.with_letters, shape.n_patterns) == (1, 1)

    def test_patterns_collapse_runs_of_a_class(self):
        shape = _stats(["AB-123", "XY-9", "John Smith", "2024-01-05", "john"]).shape
        assert shape is not None
        assert {p.pattern: p.count for p in shape.patterns} == {
            "A-9": 2,
            "Aa Aa": 1,
            "9-9-9": 1,
            "a": 1,
        }
        assert shape.n_patterns == 4

    def test_patterns_are_ranked_and_limited(self):
        shape = _stats(["a"] * 5 + ["A"] * 3 + ["9"] * 2 + ["-"], k=2).shape
        assert shape is not None
        assert [(p.pattern, p.count) for p in shape.patterns] == [("a", 5), ("A", 3)]
        assert shape.n_patterns == 4

    def test_unicode_letters_and_digits_are_classified(self):
        shape = _stats(["Éa", "é", "٣٤"]).shape
        assert shape is not None
        assert {p.pattern for p in shape.patterns} == {"Aa", "a", "9"}
        assert (shape.with_letters, shape.with_digits) == (2, 1)

    def test_control_characters_are_kept_in_the_pattern(self):
        shape = _stats(["a\x00b"]).shape
        assert shape is not None
        assert shape.patterns[0].pattern == "a\x00a"

    def test_a_profile_carries_the_shape_and_validates(self):
        profile = _categorical(["x-1", "y-2", "x-1", None, ""])
        assert profile.empty_strings == 1
        assert profile.shape is not None
        assert profile.shape.n_patterns == 1
        assert profile.shape.patterns[0].count == 3
        assert profile.lengths is not None
        assert profile.lengths.min == 0
