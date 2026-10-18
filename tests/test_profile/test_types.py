"""Semantic types come from fixed rules over a sample, in a fixed order."""

import datetime as dt
from decimal import Decimal

import pandas as pd
import polars as pl
import pytest

from dsbf.data import load
from dsbf.model.vocabulary import Confidence
from dsbf.profile import TYPE_SAMPLE_ROWS, infer_column_type, infer_column_types
from dsbf.profile.types import _date_format

pytestmark = pytest.mark.unit


def _kind(values, dtype=None, **kwargs):
    return infer_column_type(pl.Series("x", values, dtype=dtype), **kwargs)


def _days(n: int, fmt: str = "%Y-%m-%d") -> list[str]:
    start = dt.date(2023, 1, 1)
    return [(start + dt.timedelta(days=i)).strftime(fmt) for i in range(n)]


class TestNoUsableValue:
    @pytest.mark.parametrize(
        ("values", "dtype"),
        [
            ([None, None, None], pl.String),
            ([None, None, None], pl.Float64),
            ([None, None, None], pl.Int64),
            ([None, None, None], pl.Null),
            ([float("nan"), None, float("nan")], pl.Float64),
            ([], pl.String),
        ],
    )
    def test_is_unknown_whatever_dtype_held_the_nulls(self, values, dtype):
        result = _kind(values, dtype)
        assert result.semantic_type == "unknown"
        assert result.confidence is Confidence.LOW
        assert result.reason == "no values"

    def test_string_and_float_all_null_columns_agree(self):
        # The CSV reader gives String, pandas gives float64: the same answer.
        assert _kind([None] * 20, pl.String).semantic_type == (
            _kind([float("nan")] * 20, pl.Float64).semantic_type
        )

    @pytest.mark.parametrize(
        "series",
        [
            pl.Series([[1], [2, 3]]),
            pl.Series([{"a": 1}, {"a": 2}]),
            pl.Series([dt.timedelta(days=1), dt.timedelta(days=2)]),
            pl.Series([dt.time(1), dt.time(2)]),
            pl.Series([b"a", b"b"]),
        ],
        ids=["list", "struct", "duration", "time", "binary"],
    )
    def test_dtype_with_no_profile_is_unknown(self, series):
        result = infer_column_type(series)
        assert result.semantic_type == "unknown"
        assert result.physical_dtype == str(series.dtype)


class TestDtypeDecides:
    def test_boolean(self):
        result = _kind([True, False, None, True] * 5)
        assert (result.semantic_type, result.confidence) == ("boolean", Confidence.HIGH)

    @pytest.mark.parametrize(
        "series",
        [
            pl.Series([dt.date(2023, 1, i) for i in range(1, 21)]),
            pl.Series([dt.datetime(2023, 1, i, 5) for i in range(1, 21)]),
        ],
        ids=["date", "datetime"],
    )
    def test_date_dtypes(self, series):
        result = infer_column_type(series)
        assert (result.semantic_type, result.confidence) == (
            "datetime",
            Confidence.HIGH,
        )
        assert result.datetime_format is None
        assert result.sample_rows is None


class TestNumeric:
    def test_many_distinct_values(self):
        result = _kind([float(i) / 7 for i in range(500)])
        assert (result.semantic_type, result.confidence) == ("numeric", Confidence.HIGH)

    def test_two_distinct_values_is_categorical(self):
        assert _kind([0, 1] * 50).semantic_type == "categorical"

    def test_few_distinct_values_in_many_rows_is_categorical(self):
        assert _kind(list(range(20)) * 50).semantic_type == "categorical"

    def test_twenty_one_distinct_values_is_numeric(self):
        assert _kind(list(range(21)) * 50).semantic_type == "numeric"

    def test_twenty_distinct_values_in_few_rows_is_numeric(self):
        # 20 of 100 values are distinct: over the 5% line.
        assert _kind(list(range(20)) * 5).semantic_type == "numeric"

    def test_nan_is_missing_not_a_value(self):
        # Two distinct finite values, plus NaN that must not count as a third.
        assert _kind([1.5, 2.5, float("nan")] * 20).semantic_type == "categorical"

    def test_decimal_is_numeric(self):
        values = [Decimal(i) / Decimal(3) for i in range(200)]
        assert _kind(values, pl.Decimal(20, 6)).semantic_type == "numeric"

    @pytest.mark.parametrize(
        ("start", "step", "unit"),
        [(1_600_000_000, 997, "s"), (1_600_000_000_000, 99_991, "ms")],
    )
    def test_unix_timestamps(self, start, step, unit):
        result = _kind([start + i * step for i in range(200)])
        assert (result.semantic_type, result.epoch_unit) == ("datetime", unit)
        assert result.confidence is Confidence.MODERATE

    def test_few_large_values_are_not_timestamps(self):
        assert _kind([1_600_000_000 + i for i in range(4)]).semantic_type != "datetime"

    def test_infinity_counts_as_a_value(self):
        values = [float("inf")] + [float(i) / 3 for i in range(50)]
        assert _kind(values).semantic_type == "numeric"


class TestDatesInText:
    def test_iso_dates(self):
        result = _kind(_days(200))
        assert (result.semantic_type, result.datetime_format) == (
            "datetime",
            "%Y-%m-%d",
        )
        assert result.confidence is Confidence.HIGH
        assert result.sample_rows == 200

    def test_surrounding_whitespace_is_ignored(self):
        assert _kind([f" {d} " for d in _days(50)]).semantic_type == "datetime"

    @pytest.mark.parametrize(
        ("fmt", "expected"),
        [
            ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%.f"),
            ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%.f"),
            ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%.fZ"),
            ("%Y-%m-%dT%H:%M:%S+02:00", "%Y-%m-%dT%H:%M:%S%.f%z"),
            ("%Y/%m/%d", "%Y/%m/%d"),
            ("%b %d %Y", "%b %d %Y"),
            ("%B %d, %Y", "%B %d, %Y"),
            ("%d %B %Y", "%d %B %Y"),
            ("%d-%b-%Y", "%d-%b-%Y"),
        ],
    )
    def test_known_formats(self, fmt, expected):
        result = _kind(_days(60, fmt))
        assert (result.semantic_type, result.datetime_format) == ("datetime", expected)

    def test_slash_dates_with_days_over_twelve_are_day_first(self):
        result = _kind(_days(60, "%d/%m/%Y"))
        assert result.datetime_format == "%d/%m/%Y"

    def test_slash_dates_that_fit_both_orders_are_month_first(self):
        values = [f"{m:02d}/{d:02d}/2023" for m in range(1, 13) for d in range(1, 13)]
        assert _kind(values).datetime_format == "%m/%d/%Y"

    def test_year_beyond_pandas_nanosecond_range(self):
        result = _kind([f"{2200 + i}-01-07T00:00:00" for i in range(60)])
        assert result.semantic_type == "datetime"

    def test_a_format_outside_the_list_is_not_a_date(self):
        values = [f"2023.{(i % 12) + 1:02d}.{(i % 28) + 1:02d}" for i in range(100)]
        assert _kind(values * 2).semantic_type != "datetime"

    def test_compact_digits_are_not_dates(self):
        # Eight digits parse as %Y%m%d, which is why that format is not tried.
        values = pl.Series(
            "x", [f"2023{(i % 12) + 1:02d}{(i % 28) + 1:02d}" for i in range(300)]
        )
        assert infer_column_type(values).semantic_type == "identifier"

    def test_mixed_formats_parse_under_no_single_format(self):
        formats = ("%Y-%m-%d", "%d/%m/%Y", "%b %d %Y", "%Y/%m/%d")
        columns = [_days(25, fmt) for fmt in formats]
        mixed = [day for row in zip(*columns, strict=True) for day in row] * 4
        assert _kind(mixed).semantic_type == "categorical"

    def test_ninety_percent_is_not_enough_and_ninety_one_is(self):
        junk = "not a date"
        at_line = _days(90) + [junk] * 10
        over_line = _days(91) + [junk] * 9
        assert _kind(at_line).semantic_type != "datetime"
        over = _kind(over_line)
        assert (over.semantic_type, over.confidence) == (
            "datetime",
            Confidence.MODERATE,
        )

    def test_confirmation_reads_the_whole_column(self):
        # A sample of clean dates from a column that is mostly something else.
        sample = pl.Series(_days(100))
        column = pl.Series(_days(100) + ["n/a"] * 400)
        assert _date_format(sample, column, column.len()) is None
        assert _date_format(sample, pl.Series(_days(100) * 5), 500) == (
            "%Y-%m-%d",
            1.0,
        )


class TestStringShapes:
    def test_period_labels_are_categorical(self):
        values = [f"FY{2000 + i % 20}" for i in range(200)]
        result = _kind(values)
        assert (result.semantic_type, result.reason) == ("categorical", "period labels")

    def test_period_labels_over_a_hundred_distinct_are_not_labels(self):
        values = [f"'{i:02d}-{j:02d}" for i in range(20) for j in range(20)]
        assert _kind(values).reason != "period labels"

    def test_uuids_are_identifiers(self):
        values = [f"{i:08x}-0000-4000-8000-{i:012x}" for i in range(40)] * 3
        assert _kind(values).semantic_type == "identifier"

    def test_nearly_all_distinct_strings_are_identifiers(self):
        assert _kind([f"user {i}" for i in range(100)]).semantic_type == "identifier"

    def test_repeated_long_strings_are_text(self):
        sentence = "a sentence that is a good deal longer than thirty characters"
        values = [f"{sentence} {i % 3}" for i in range(300)]
        assert _kind(values).semantic_type == "text"

    def test_repeated_short_strings_are_categorical(self):
        result = _kind(["red", "green", "blue"] * 100)
        assert (result.semantic_type, result.confidence) == (
            "categorical",
            Confidence.MODERATE,
        )

    def test_categorical_dtype_is_read_as_its_strings(self):
        values = pl.Series("x", ["red", "green", "blue"] * 100, dtype=pl.Categorical)
        result = infer_column_type(values)
        assert result.semantic_type == "categorical"
        assert result.physical_dtype == "Categorical"


class TestSampling:
    def test_text_rules_read_at_most_sample_rows_values(self):
        values = [f"item {i % 40}" for i in range(50_000)]
        assert _kind(values).sample_rows == TYPE_SAMPLE_ROWS
        assert _kind(values, sample_rows=64).sample_rows == 64

    def test_a_short_column_is_read_whole(self):
        assert _kind(["a", "b"] * 30).sample_rows == 60

    def test_the_sample_is_the_same_every_time(self):
        values = [
            f"{(i * 7919) % 100_003}" if i % 3 else _days(1)[0] for i in range(9_000)
        ]
        assert _kind(values) == _kind(values)

    def test_nulls_are_not_part_of_the_sample(self):
        values = [None] * 5_000 + _days(200)
        result = _kind(values)
        assert (result.semantic_type, result.sample_rows) == ("datetime", 200)

    @pytest.mark.parametrize("rows", [0, -1, 2.5, True, "10"])
    def test_sample_rows_must_be_a_positive_integer(self, rows):
        with pytest.raises(ValueError, match="sample_rows"):
            _kind(["a"], sample_rows=rows)


class TestConfidence:
    def test_fewer_than_ten_values_are_never_firm(self):
        assert _kind([True, False, True]).confidence is Confidence.LOW
        assert _kind(_days(9)).confidence is Confidence.LOW
        assert _kind(_days(10)).confidence is Confidence.HIGH

    def test_confirmed_dates_below_ninety_nine_percent_are_moderate(self):
        result = _kind(_days(96) + ["junk"] * 4)
        assert (result.semantic_type, result.confidence) == (
            "datetime",
            Confidence.MODERATE,
        )


class TestDataset:
    def test_every_column_in_order(self):
        dataset = load(
            pl.DataFrame(
                {
                    "when": _days(30),
                    "n": [i / 7 for i in range(30)],
                    "ok": [i % 2 == 0 for i in range(30)],
                    "empty": [None] * 30,
                }
            )
        )
        types = infer_column_types(dataset)
        assert list(types) == ["when", "n", "ok", "empty"]
        assert [t.semantic_type for t in types.values()] == [
            "datetime",
            "numeric",
            "boolean",
            "unknown",
        ]

    def test_pandas_and_polars_input_agree(self):
        data = {
            "when": _days(60),
            "n": [i / 7 for i in range(60)],
            "ok": [i % 2 == 0 for i in range(60)],
            "label": ["red", "green", "blue"] * 20,
            "empty": [float("nan")] * 60,
        }
        from_pandas = infer_column_types(load(pd.DataFrame(data)))
        from_polars = infer_column_types(load(pl.DataFrame(data)))
        assert {k: v.semantic_type for k, v in from_pandas.items()} == {
            k: v.semantic_type for k, v in from_polars.items()
        }

    def test_an_empty_dataset_has_only_unknown_columns(self):
        dataset = load(
            pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Int64, "b": pl.String})
        )
        assert {t.semantic_type for t in infer_column_types(dataset).values()} == {
            "unknown"
        }
