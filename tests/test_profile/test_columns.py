"""Counts and numeric statistics are per column, over that column's own values."""

import math
import subprocess
import sys
import textwrap
from decimal import Decimal

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pydantic import TypeAdapter
from scipy.stats import entropy as scipy_entropy

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


def _by_name(frame: pl.DataFrame, **kwargs):
    return {p.name: p for p in _profile(frame, **kwargs)}


class TestEntropy:
    def test_matches_scipy_over_the_whole_distribution(self):
        # Fifteen distinct values: beyond the ten top values the profile lists.
        values = [chr(97 + i) for i in range(15) for _ in range(i + 1)]
        counts = np.arange(1, 16)

        profile = _one(values)

        assert isinstance(profile, CategoricalProfile)
        assert len(profile.top_values) == 10
        assert profile.entropy_bits == pytest.approx(scipy_entropy(counts, base=2))

    def test_text_columns_have_it_too(self):
        profile = _one(["a b c", "a b c", "d e f", "g h i"] * 5 + ["j k l"])

        assert isinstance(profile, TextProfile | CategoricalProfile)
        assert profile.entropy_bits == pytest.approx(
            scipy_entropy([10, 5, 5, 1], base=2)
        )

    def test_a_boolean_column_has_the_entropy_of_its_split(self):
        profile = _one([True, True, True, False, None])

        assert isinstance(profile, BooleanProfile)
        assert profile.entropy_bits == pytest.approx(scipy_entropy([3, 1], base=2))

    def test_one_value_is_a_positive_zero(self):
        profile = _one(["same"] * 6)

        assert isinstance(profile, CategoricalProfile)
        assert profile.entropy_bits == 0.0
        assert repr(profile.entropy_bits) == "0.0"  # not -0.0

    def test_a_uniform_column_has_log2_of_its_values(self):
        profile = _one(list("abcdefgh") * 3)

        assert isinstance(profile, CategoricalProfile)
        assert profile.entropy_bits == pytest.approx(3.0)

    def test_nulls_are_not_a_value(self):
        profile = _one(["a", "b", "a", "b", None, None, None, None])

        assert isinstance(profile, CategoricalProfile)
        assert profile.entropy_bits == pytest.approx(1.0)

    def test_a_boolean_with_one_value_is_a_positive_zero(self):
        profile = _one([True, None, True], dtype=pl.Boolean)

        assert isinstance(profile, BooleanProfile)
        assert profile.entropy_bits == 0.0
        assert repr(profile.entropy_bits) == "0.0"  # not -0.0

    def test_does_not_depend_on_how_many_top_values_are_listed(self):
        values = list("aabbbcccc")

        assert _one(values, top_k=0).entropy_bits == _one(values).entropy_bits

    def test_numeric_columns_have_none(self):
        profile = _one([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])

        assert isinstance(profile, NumericProfile)
        assert not hasattr(profile, "entropy_bits")


class TestContentHash:
    FRAME = pl.DataFrame(
        {
            "a": [1, 2, 3, None, 5, 6, 7, 8],
            "same": [1, 2, 3, None, 5, 6, 7, 8],
            "reordered": [2, 1, 3, None, 5, 6, 7, 8],
            "changed": [1, 2, 3, None, 5, 6, 7, 9],
            "no_null": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    def test_equal_columns_hash_equal_and_other_columns_do_not(self):
        hashes = {name: p.content_hash for name, p in _by_name(self.FRAME).items()}

        assert hashes["a"] == hashes["same"]
        assert len({hashes[n] for n in ("a", "reordered", "changed", "no_null")}) == 4

    def test_a_hash_is_sixteen_hex_digits_even_for_no_rows_or_all_nulls(self):
        empty = _one([], dtype=pl.Int64)
        nulls = _one([None, None, None], dtype=pl.Int64)

        for profile in (empty, nulls, *_profile(self.FRAME)):
            assert profile.content_hash is not None
            assert len(profile.content_hash) == 16
            int(profile.content_hash, 16)

    def test_the_hash_follows_row_order(self):
        forward = _one(list("abcdef"))
        backward = _one(list("fedcba"))

        assert forward.content_hash != backward.content_hash

    def test_the_hash_ignores_the_dtype_so_the_dtype_must_be_compared(self):
        frame = pl.DataFrame(
            {
                "ints": pl.Series([1, 2, 3, 4], dtype=pl.Int64),
                "narrow": pl.Series([1, 2, 3, 4], dtype=pl.Int32),
                "dates": pl.Series([1, 2, 3, 4], dtype=pl.Int64).cast(pl.Date),
            }
        )
        by_name = _by_name(frame)

        assert by_name["ints"].content_hash == by_name["dates"].content_hash
        assert by_name["ints"].physical_dtype != by_name["dates"].physical_dtype

    def test_negative_zero_and_nan_hash_as_equal_values(self):
        frame = pl.DataFrame(
            {
                "x": [0.0, float("nan"), 1.0, 2.0],
                "y": [-0.0, float("nan"), 1.0, 2.0],
            }
        )
        by_name = _by_name(frame)

        assert by_name["x"].content_hash == by_name["y"].content_hash

    def test_a_categorical_hashes_by_its_values_not_its_codes(self):
        values = ["x", "y", None, "z", "x", "y"]
        frame = pl.DataFrame(
            {
                "plain": pl.Series(values, dtype=pl.Categorical),
                "recoded": pl.Series(
                    values, dtype=pl.Categorical
                ),  # built again, same strings
                "string": pl.Series(values),
            }
        )
        by_name = _by_name(frame)

        assert by_name["plain"].content_hash == by_name["recoded"].content_hash

    def test_an_enum_whose_codes_count_up_is_not_hashed_to_zero(self):
        enum = pl.Enum(["a", "b", "c"])
        profile = _one(["a", "b", "c", "a"], dtype=enum)

        assert profile.content_hash != "0" * 16

    def test_an_object_column_has_no_hash(self):
        profile = _one([object(), "x", None], dtype=pl.Object)

        assert profile.content_hash is None

    def test_nested_and_other_dtypes_hash(self):
        frame = pl.DataFrame(
            {
                "list": [[1], [2], None],
                "struct": [{"a": 1}, {"a": 2}, None],
                "bytes": [b"a", b"b", None],
                "decimal": pl.Series([Decimal("1.5"), None, Decimal("2")]),
            }
        )

        assert all(p.content_hash for p in _profile(frame))

    def test_the_digest_is_the_same_in_another_process(self):
        script = textwrap.dedent(
            """
            import polars as pl
            from dsbf.data import load
            from dsbf.profiling import profile_columns

            frame = pl.DataFrame(
                {"a": [1, None, 3], "s": ["x", "y", None], "f": [0.5, -0.0, 2.5]}
            )
            print([p.content_hash for p in profile_columns(load(frame))])
            """
        )
        runs = {
            subprocess.run(
                [sys.executable, "-I", "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env={"PYTHONHASHSEED": seed},
            ).stdout
            for seed in ("1", "2", "random")
        }

        assert len(runs) == 1
        frame = pl.DataFrame(
            {"a": [1, None, 3], "s": ["x", "y", None], "f": [0.5, -0.0, 2.5]}
        )
        assert runs == {str([p.content_hash for p in _profile(frame)]) + "\n"}


class TestEncodingDamage:
    FRAME = pl.DataFrame(
        {
            "clean": ["a", "b", "c", "d"],
            "replaced": ["a\ufffd", "b", "\ufffd\ufffd", None],
            "nulled": ["a\x00b", "c", "d", "e\x00"],
            "both": ["\ufffd\x00", "x", "y", "z"],
            "number": [1.5, 2.5, 3.5, 4.5],
            "bad\ufffdname": ["p", "q", "r", "s"],
            "nul\x00name": [1, 2, 3, 4],
        }
    )

    def test_counts_are_of_values_not_characters(self):
        by_name = _by_name(self.FRAME)

        assert (
            by_name["clean"].values_with_replacement_char,
            by_name["clean"].values_with_nul,
        ) == (0, 0)
        assert by_name["replaced"].values_with_replacement_char == 2
        assert by_name["replaced"].values_with_nul == 0
        assert by_name["nulled"].values_with_nul == 2
        assert by_name["nulled"].values_with_replacement_char == 0
        assert by_name["both"].values_with_replacement_char == 1
        assert by_name["both"].values_with_nul == 1

    def test_columns_that_are_not_strings_have_none(self):
        by_name = _by_name(self.FRAME)

        for name in ("number", "nul\x00name"):
            assert by_name[name].values_with_replacement_char is None
            assert by_name[name].values_with_nul is None

    def test_a_damaged_name_is_flagged_whatever_the_column_holds(self):
        by_name = _by_name(self.FRAME)

        flagged = {name for name, p in by_name.items() if p.name_encoding_damage}
        assert flagged == {"bad\ufffdname", "nul\x00name"}

    def test_every_string_column_is_counted_whatever_its_semantic_type(self):
        values = [f"id\ufffd{i}" for i in range(50)]
        frame = pl.DataFrame(
            {
                "long_unique": values,
                "category": pl.Series(["a\x00", "b"] * 25, dtype=pl.Categorical),
                "enum": pl.Series(["a\x00", "b"] * 25, dtype=pl.Enum(["a\x00", "b"])),
            }
        )
        by_name = _by_name(frame)

        assert by_name["long_unique"].values_with_replacement_char == 50
        assert by_name["category"].values_with_nul == 25
        assert by_name["enum"].values_with_nul == 25

    def test_a_utf16_file_read_lossily_shows_both_marks(self, tmp_path):
        path = tmp_path / "wide.csv"
        rows = "".join(f"Anna{i},Zürich,{i}\n" for i in range(50))
        path.write_bytes(("name,city,n\n" + rows).encode("utf-16"))
        frame = pl.read_csv(path, encoding="utf8-lossy")

        profiles = _profile(frame)

        assert all(p.name_encoding_damage for p in profiles)
        strings = [p for p in profiles if p.values_with_nul is not None]
        assert strings
        assert all(
            p.values_with_nul and p.values_with_replacement_char is not None
            for p in strings
        )
        assert any(p.values_with_replacement_char for p in strings)


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
