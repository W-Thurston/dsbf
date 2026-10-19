"""Counts and numeric statistics are per column, over that column's own values."""

import math
from decimal import Decimal

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pydantic import TypeAdapter

from dsbf.data import load
from dsbf.model.profile import (
    BooleanProfile,
    CategoricalProfile,
    ColumnProfile,
    DatetimeProfile,
    IdentifierProfile,
    NumericProfile,
    TextProfile,
    UnknownProfile,
)
from dsbf.model.vocabulary import Confidence
from dsbf.profiling import (
    DEFAULT_QUANTILES,
    ColumnType,
    infer_column_types,
    profile_columns,
)
from tests.fixtures.adversarial import build_adversarial_frame

pytestmark = pytest.mark.unit


def _profile(frame: pl.DataFrame | pd.DataFrame, **kwargs):
    return profile_columns(load(frame), **kwargs)


def _one(values, dtype=None, **kwargs):
    (profile,) = _profile(pl.DataFrame({"x": pl.Series(values, dtype=dtype)}), **kwargs)
    return profile


def _numeric(values, dtype=None, **kwargs) -> NumericProfile:
    """Profile a column as numeric, however few distinct values it has."""
    frame = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
    dataset = load(frame)
    forced = ColumnType("numeric", Confidence.HIGH, str(frame.schema["x"]), "forced")
    (profile,) = profile_columns(dataset, {"x": forced}, **kwargs)
    assert isinstance(profile, NumericProfile)
    return profile


class TestCounts:
    def test_nulls_and_distinct_count_the_columns_own_values(self):
        profile = _one([1, 2, 2, None, None, 3])
        assert (profile.counts.rows, profile.counts.nulls, profile.counts.distinct) == (
            6,
            2,
            3,
        )
        assert profile.counts.distinct_exact

    def test_a_pandas_nan_is_null_and_a_polars_nan_is_not(self):
        from_pandas = _profile(pd.DataFrame({"x": [np.nan, *range(30)]}))[0]
        native = _numeric([float("nan"), 1.0, 3.0], pl.Float64)
        assert isinstance(from_pandas, NumericProfile)
        assert from_pandas.counts.nulls == 1
        assert from_pandas.non_finite == 0
        assert isinstance(native, NumericProfile)
        assert native.counts.nulls == 0
        assert native.non_finite == 1

    def test_profiles_follow_column_order_and_carry_the_dtype(self):
        frame = pl.DataFrame({"b": [1, 2, 3], "a": ["x", "y", "x"]})
        profiles = _profile(frame)
        assert [(p.name, p.position) for p in profiles] == [("b", 0), ("a", 1)]
        assert [p.physical_dtype for p in profiles] == ["Int64", "String"]

    def test_a_sampled_dataset_counts_the_rows_it_holds(self):
        dataset = load(pl.DataFrame({"x": list(range(100))})).sample(10)
        (profile,) = profile_columns(dataset)
        assert profile.counts.rows == 10
        assert profile.counts.distinct == 10

    def test_a_frame_with_no_rows_has_zero_counts(self):
        profile = _one([], pl.Float64)
        assert isinstance(profile, UnknownProfile)
        assert (profile.counts.rows, profile.counts.nulls, profile.counts.distinct) == (
            0,
            0,
            0,
        )

    def test_an_all_null_column_is_unknown(self):
        profile = _one([None, None, None], pl.Float64)
        assert isinstance(profile, UnknownProfile)
        assert (profile.counts.nulls, profile.counts.distinct) == (3, 0)

    def test_an_object_column_is_counted_in_python(self):
        marker = object()
        frame = pl.DataFrame({"x": pl.Series([marker, marker, None], dtype=pl.Object)})
        (profile,) = _profile(frame)
        assert isinstance(profile, UnknownProfile)
        assert (profile.counts.nulls, profile.counts.distinct) == (1, 1)

    def test_nested_and_binary_columns_have_counts(self):
        frame = pl.DataFrame(
            {"l": [[1], [1], None], "s": [{"a": 1}, {"a": 2}, None], "b": [b"a"] * 3}
        )
        counts = {p.name: (p.counts.nulls, p.counts.distinct) for p in _profile(frame)}
        assert counts == {"l": (1, 1), "s": (1, 2), "b": (0, 1)}


class TestNumericStatistics:
    def test_match_pandas_on_the_columns_own_values(self):
        values = [1.0, 2.0, 2.0, 3.0, 10.0, 0.0, 0.0, 7.0, None, 4.5]
        profile = _numeric(values)
        series = pd.Series(values).dropna()
        stats = profile.stats
        assert stats is not None
        assert stats.min == series.min()
        assert stats.max == series.max()
        assert stats.mean == pytest.approx(series.mean())
        assert stats.std == pytest.approx(series.std())
        assert stats.skewness == pytest.approx(series.skew())
        assert stats.kurtosis == pytest.approx(series.kurt())
        assert stats.quantile_method == "exact"
        assert [q.p for q in stats.quantiles] == list(DEFAULT_QUANTILES)
        for quantile in stats.quantiles:
            assert quantile.value == pytest.approx(series.quantile(quantile.p))
        assert profile.zeros == 2

    def test_nulls_in_one_column_do_not_shrink_another(self):
        frame = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [None, None, None, 9.0]})
        a, b = _profile(frame)
        assert isinstance(a, NumericProfile)
        assert isinstance(b, NumericProfile)
        assert a.stats is not None
        assert a.stats.mean == 2.5
        assert b.stats is not None
        assert b.stats.mean == 9.0

    def test_infinities_are_counted_and_left_out_of_the_statistics(self):
        profile = _numeric([1.0, 2.0, 3.0, 4.0, float("inf"), -math.inf, float("nan")])
        assert profile.non_finite == 3
        assert profile.stats is not None
        assert (profile.stats.min, profile.stats.max, profile.stats.mean) == (1, 4, 2.5)
        assert profile.stats.non_finite_stats == ()

    def test_only_non_finite_values_leave_no_statistics(self):
        profile = _numeric([float("inf"), float("nan")], pl.Float64)
        assert profile.non_finite == 2
        assert profile.stats is None

    def test_negative_zero_is_a_zero(self):
        assert _numeric([0.0, -0.0, 1.0]).zeros == 2

    def test_a_single_value_has_undefined_spread_named_in_the_stats(self):
        stats = _numeric([5.0]).stats
        assert stats is not None
        assert (stats.min, stats.max, stats.mean) == (5, 5, 5)
        assert stats.std is None
        assert set(stats.non_finite_stats) == {"std", "skewness", "kurtosis"}

    def test_a_constant_column_has_zero_spread_and_undefined_shape(self):
        stats = _numeric([4.0, 4.0, 4.0, 4.0, 4.0]).stats
        assert stats is not None
        assert stats.std == 0
        assert set(stats.non_finite_stats) == {"skewness", "kurtosis"}

    def test_two_values_have_no_skewness(self):
        stats = _numeric([1.0, 2.0]).stats
        assert stats is not None
        assert stats.std == pytest.approx(math.sqrt(0.5))
        assert "skewness" in stats.non_finite_stats

    def test_integers_beyond_float_range_do_not_overflow(self):
        stats = _numeric([2**64 - 1, 1, 5], pl.UInt64).stats
        assert stats is not None
        assert stats.max == float(2**64 - 1)
        assert stats.mean == pytest.approx((2**64 + 5) / 3 - 1 / 3, rel=1e-12)

    def test_decimals_are_summarised_as_floats(self):
        values = [Decimal("1.5"), Decimal("2.5"), None]
        stats = _numeric(values, pl.Decimal(10, 2)).stats
        assert stats is not None
        assert stats.mean == 2.0

    def test_quantiles_are_chosen_by_the_caller(self):
        stats = _numeric(list(range(101)), quantiles=(0.1, 0.9)).stats
        assert stats is not None
        assert [(q.p, q.value) for q in stats.quantiles] == [(0.1, 10), (0.9, 90)]

    def test_no_quantiles_leave_the_method_unset(self):
        stats = _numeric([1.0, 2.0, 3.0], quantiles=()).stats
        assert stats is not None
        assert stats.quantiles == ()
        assert stats.quantile_method is None

    @pytest.mark.parametrize("quantiles", [(0.5, 0.1), (0.5, 0.5), (-0.1,), (1.1,)])
    def test_bad_quantiles_are_rejected(self, quantiles):
        with pytest.raises(ValueError, match="quantiles"):
            _one([1.0, 2.0], quantiles=quantiles)


class TestOtherSemanticTypes:
    def test_text_columns_count_empty_strings(self):
        profile = _one(["a", "", "", "b", None] * 4)
        assert isinstance(profile, CategoricalProfile)
        assert profile.empty_strings == 8

    def test_a_categorical_dtype_counts_empty_strings(self):
        profile = _one(["a", "", "b", "a"] * 5, pl.Categorical)
        assert isinstance(profile, CategoricalProfile)
        assert profile.empty_strings == 5

    def test_numeric_looking_categories_have_no_empty_strings(self):
        profile = _one([1, 2, 1, 2, 1, 2])
        assert isinstance(profile, CategoricalProfile)
        assert profile.empty_strings == 0

    def test_booleans_split_into_true_and_false(self):
        profile = _one([True, False, True, None, True])
        assert isinstance(profile, BooleanProfile)
        assert (profile.true_count, profile.false_count) == (3, 1)

    def test_long_strings_are_text(self):
        profile = _one([("word " * 10) + str(i % 3) for i in range(40)])
        assert isinstance(profile, TextProfile)

    def test_dates_and_identifiers_keep_the_common_fields(self):
        frame = pl.DataFrame(
            {
                "when": pl.date_range(
                    pl.date(2024, 1, 1), pl.date(2024, 1, 20), eager=True
                ),
                "key": [f"{i:08x}" for i in range(20)],
            }
        )
        when, key = _profile(frame)
        assert isinstance(when, DatetimeProfile)
        assert when.counts.distinct == 20
        assert isinstance(key, IdentifierProfile)

    def test_confidence_and_sample_size_come_from_the_type(self):
        frame = pl.DataFrame({"x": [f"2024-01-{d:02d}" for d in range(1, 29)]})
        dataset = load(frame)
        types = infer_column_types(dataset)
        (profile,) = profile_columns(dataset, types)
        assert profile.semantic_confidence == types["x"].confidence
        assert profile.semantic_sample_rows == types["x"].sample_rows == 28


class TestArguments:
    def test_types_are_used_as_given(self):
        dataset = load(pl.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}))
        types = infer_column_types(dataset)
        assert isinstance(profile_columns(dataset, types)[0], NumericProfile)

    def test_types_must_cover_the_columns(self):
        dataset = load(pl.DataFrame({"x": [1, 2], "y": [3, 4]}))
        types = infer_column_types(dataset)
        with pytest.raises(ValueError, match="exactly the dataset's columns"):
            profile_columns(dataset, {"x": types["x"]})

    def test_a_numeric_type_needs_a_numeric_column(self):
        frame = pl.DataFrame({"x": list(range(10)), "s": list("abcdefghij")})
        dataset = load(frame)
        types = infer_column_types(dataset)
        with pytest.raises(ValueError, match="typed numeric"):
            profile_columns(dataset, {**types, "s": types["x"]})

    def test_duplicate_looking_names_and_odd_names_are_fine(self):
        frame = pl.DataFrame({"0:nulls": [1.0, 2.0], "": [3.0, 4.0], " ": [5.0, 6.0]})
        names = [p.name for p in _profile(frame)]
        assert names == ["0:nulls", "", " "]


def test_profiles_round_trip_through_the_contract():
    frame = pl.DataFrame(
        {
            "n": [1.0, 2.0, None, float("inf")],
            "s": ["a", "", "b", None],
            "b": [True, False, None, True],
        }
    )
    adapter = TypeAdapter(ColumnProfile)
    for profile in _profile(frame):
        assert adapter.validate_json(profile.model_dump_json()) == profile


def test_the_adversarial_frame_profiles_every_column():
    frame = build_adversarial_frame()
    profiles = profile_columns(load(frame))
    assert [p.name for p in profiles] == [str(c) for c in frame.columns]
    by_name = {p.name: p for p in profiles}
    assert all(p.counts.rows == len(frame) for p in profiles)
    assert by_name["all_null"].counts.nulls == len(frame)
    assert by_name["all_nan_float"].counts.nulls == len(frame)
    assert isinstance(by_name["all_nan_float"], UnknownProfile)
    inf_col = by_name["inf_col"]
    assert isinstance(inf_col, NumericProfile)
    assert inf_col.non_finite > 0
    assert inf_col.stats is not None
    assert math.isfinite(inf_col.stats.mean or 0.0)


def test_the_adversarial_frame_profiles_to_strict_json_and_back():
    frame = build_adversarial_frame()
    profiles = profile_columns(load(frame))
    adapter = TypeAdapter(ColumnProfile)
    for profile in profiles:
        text = profile.model_dump_json()
        assert "NaN" not in text
        assert "Infinity" not in text
        assert adapter.validate_json(text) == profile
        if isinstance(profile, NumericProfile) and profile.histogram:
            assert (
                profile.histogram.total == profile.counts.non_null - profile.non_finite
            )
