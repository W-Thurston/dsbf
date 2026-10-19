"""Range, time zone, frequency and gaps of datetime columns."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import polars as pl
import pytest

from dsbf.data import load
from dsbf.model.profile import DatetimeProfile
from dsbf.model.vocabulary import Confidence
from dsbf.profiling import ColumnType, profile_columns
from dsbf.profiling.temporal import datetime_stats

pytestmark = pytest.mark.unit


def _type(**fields) -> ColumnType:
    return ColumnType("datetime", Confidence.HIGH, "?", "test", **fields)


def _stats(values, dtype=None, **type_fields):
    series = pl.Series("t", values, dtype=dtype)
    return datetime_stats(series, _type(**type_fields))


def _days(start: datetime, count: int, step=timedelta(days=1)) -> list[datetime]:
    return [start + i * step for i in range(count)]


class TestRange:
    def test_a_datetime_column_has_its_min_and_max(self):
        stats = _stats(_days(datetime(2024, 1, 1), 10))
        assert (stats.min, stats.max) == (datetime(2024, 1, 1), datetime(2024, 1, 10))

    def test_a_date_column_reads_as_midnight(self):
        stats = _stats([date(2024, 3, 1), date(2024, 3, 5), None])
        assert (stats.min, stats.max) == (datetime(2024, 3, 1), datetime(2024, 3, 5))

    def test_nulls_are_ignored_and_an_all_null_column_has_nothing(self):
        assert _stats([None, None], pl.Datetime("us")) == datetime_stats(
            pl.Series("t", [], dtype=pl.Datetime("us")), _type()
        )
        assert _stats([None, datetime(2024, 1, 1)]).min == datetime(2024, 1, 1)

    def test_a_range_past_year_9999_is_not_represented(self):
        series = pl.Series("t", [datetime(2020, 1, 1)] * 2 + [datetime(2020, 1, 2)])
        far = series.dt.offset_by("100000y")
        stats = datetime_stats(far, _type())
        assert (stats.min, stats.max) == (None, None)
        assert stats.largest_gap_seconds == 86_400.0

    def test_text_is_parsed_with_the_inferred_format(self):
        stats = _stats(["03/01/2024", "03/05/2024", "bad"], datetime_format="%m/%d/%Y")
        assert (stats.min, stats.max) == (datetime(2024, 3, 1), datetime(2024, 3, 5))

    def test_text_without_a_format_has_no_range(self):
        assert _stats(["2024-01-01", "2024-01-02"]) == datetime_stats(
            pl.Series("t", []), _type()
        )

    def test_unix_timestamps_are_read_in_their_unit(self):
        seconds = _stats([1_700_000_000, 1_700_086_400], epoch_unit="s")
        millis = _stats([1_700_000_000_000, 1_700_086_400_000], epoch_unit="ms")
        assert seconds.min == millis.min == datetime(2023, 11, 14, 22, 13, 20)
        assert seconds.largest_gap_seconds == 86_400.0

    def test_a_numeric_column_without_a_unit_has_no_range(self):
        assert _stats([1, 2, 3]).min is None


class TestTimezone:
    def test_a_naive_column_has_no_timezone(self):
        assert _stats(_days(datetime(2024, 1, 1), 5)).timezone is None

    def test_an_aware_column_keeps_its_zone(self):
        values = _days(datetime(2024, 1, 1), 5)
        series = pl.Series("t", values).dt.replace_time_zone("Europe/Paris")
        stats = datetime_stats(series, _type())
        assert stats.timezone == "Europe/Paris"
        assert stats.min is not None and stats.min.tzinfo is not None
        assert stats.max is not None and stats.max.tzinfo is not None

    def test_offsets_in_text_are_converted_to_utc(self):
        stats = _stats(
            ["2024-01-01T00:00:00+02:00", "2024-01-01T05:00:00-05:00"],
            datetime_format="%Y-%m-%dT%H:%M:%S%.f%z",
        )
        assert stats.timezone == "UTC"
        assert stats.min == datetime(2023, 12, 31, 22, tzinfo=UTC)

    def test_a_trailing_z_means_utc(self):
        stats = _stats(
            ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
            datetime_format="%Y-%m-%dT%H:%M:%S%.fZ",
        )
        assert stats.timezone == "UTC"
        assert stats.max == datetime(2024, 1, 2, tzinfo=UTC)


class TestFrequency:
    @pytest.mark.parametrize(
        ("step", "alias"),
        [
            (timedelta(seconds=1), "s"),
            (timedelta(minutes=1), "min"),
            (timedelta(minutes=15), "15min"),
            (timedelta(hours=1), "h"),
            (timedelta(hours=6), "6h"),
            (timedelta(days=1), "D"),
            (timedelta(days=2), "2D"),
            (timedelta(days=7), "W"),
            (timedelta(days=14), "2W"),
            (timedelta(milliseconds=250), "250ms"),
        ],
    )
    def test_fixed_steps_have_an_alias(self, step, alias):
        stats = _stats(_days(datetime(2024, 1, 1), 30, step))
        assert stats.inferred_frequency == alias
        assert stats.n_gaps == 0

    @pytest.mark.parametrize(
        ("pandas_freq", "alias"),
        [("D", "D"), ("h", "h"), ("min", "min"), ("W", "W"), ("2D", "2D")],
    )
    def test_regular_series_agree_with_pandas(self, pandas_freq, alias):
        index = pd.date_range("2024-01-03", periods=40, freq=pandas_freq)
        assert pd.infer_freq(index) is not None
        stats = _stats(index.to_pydatetime().tolist())
        assert stats.inferred_frequency == alias

    @pytest.mark.parametrize(
        ("pandas_freq", "alias"),
        [
            ("MS", "MS"),
            ("ME", "ME"),
            ("3MS", "3MS"),
            ("3ME", "3ME"),
            ("12MS", "12MS"),
            ("YE", "12ME"),
        ],
    )
    def test_calendar_steps_are_counted_in_months(self, pandas_freq, alias):
        index = pd.date_range("2020-01-01", periods=30, freq=pandas_freq)
        stats = _stats(index.to_pydatetime().tolist())
        assert stats.inferred_frequency == alias
        assert stats.n_gaps == 0

    def test_month_ends_are_found_through_february(self):
        index = pd.date_range("2023-11-30", periods=8, freq="ME")
        assert _stats(index.to_pydatetime().tolist()).inferred_frequency == "ME"

    def test_irregular_spacing_has_no_frequency_but_a_largest_gap(self):
        base = datetime(2024, 1, 1)
        offsets = [0, 1, 3, 7, 8, 15, 30, 31, 60, 100]
        stats = _stats([base + timedelta(days=d) for d in offsets])
        assert stats.inferred_frequency is None
        assert stats.n_gaps is None
        assert stats.largest_gap_seconds == 40 * 86_400.0

    def test_two_instants_have_a_gap_but_no_frequency(self):
        stats = _stats([datetime(2024, 1, 1), datetime(2024, 1, 3)])
        assert stats.inferred_frequency is None
        assert stats.largest_gap_seconds == 2 * 86_400.0

    def test_one_instant_has_no_gaps_at_all(self):
        stats = _stats([datetime(2024, 1, 1)] * 5)
        assert stats.min == stats.max == datetime(2024, 1, 1)
        assert (stats.largest_gap_seconds, stats.n_gaps) == (None, None)

    def test_repeated_timestamps_are_not_steps(self):
        values = [d for d in _days(datetime(2024, 1, 1), 20) for _ in range(3)]
        stats = _stats(values)
        assert stats.inferred_frequency == "D"
        assert stats.n_gaps == 0

    def test_order_does_not_matter(self):
        values = _days(datetime(2024, 1, 1), 20)
        shuffled = values[10:] + values[:10]
        assert _stats(shuffled) == _stats(values)


class TestGaps:
    def test_a_missing_stretch_is_one_gap(self):
        days = _days(datetime(2024, 1, 1), 40)
        stats = _stats([*days[:10], *days[15:]])
        assert stats.inferred_frequency == "D"
        assert stats.n_gaps == 1
        assert stats.largest_gap_seconds == 6 * 86_400.0

    def test_a_single_missing_step_is_a_gap(self):
        days = _days(datetime(2024, 1, 1), 30)
        stats = _stats([d for i, d in enumerate(days) if i != 10])
        assert (stats.n_gaps, stats.largest_gap_seconds) == (1, 2 * 86_400.0)

    def test_missing_months_are_gaps(self):
        index = pd.date_range("2020-01-01", periods=24, freq="MS")
        kept = [d for i, d in enumerate(index.to_pydatetime()) if i not in (5, 6, 7)]
        stats = _stats(kept)
        assert stats.inferred_frequency == "MS"
        assert stats.n_gaps == 1

    def test_jittered_timestamps_have_no_frequency_and_so_no_gaps(self):
        # No single step covers half the intervals, so none is claimed.
        base = datetime(2024, 1, 1)
        minutes = [0, 10, 21, 29, 40, 51, 60, 70, 80, 91, 99, 110]
        stats = _stats([base + timedelta(minutes=m) for m in minutes])
        assert (stats.inferred_frequency, stats.n_gaps) == (None, None)
        assert stats.largest_gap_seconds == 11 * 60.0


class TestInProfile:
    def test_a_datetime_column_is_profiled(self):
        frame = pl.DataFrame({"t": _days(datetime(2024, 1, 1), 50)})
        (profile,) = profile_columns(load(frame))
        assert isinstance(profile, DatetimeProfile)
        assert profile.min == datetime(2024, 1, 1)
        assert profile.inferred_frequency == "D"
        assert profile.n_gaps == 0

    def test_a_text_date_column_is_profiled_through_its_format(self, tmp_path):
        path = tmp_path / "d.csv"
        dates = [d.strftime("%m/%d/%Y") for d in _days(datetime(2024, 1, 13), 40)]
        path.write_text("when\n" + "\n".join(dates) + "\n")
        (profile,) = profile_columns(load(path))
        assert isinstance(profile, DatetimeProfile)
        assert (profile.min, profile.max) == (
            datetime(2024, 1, 13),
            datetime(2024, 2, 21),
        )
        assert profile.inferred_frequency == "D"

    def test_an_epoch_column_is_profiled_through_its_unit(self):
        start = 1_700_000_000
        frame = pl.DataFrame({"t": [start + 3600 * i for i in range(40)]})
        (profile,) = profile_columns(load(frame))
        assert isinstance(profile, DatetimeProfile)
        assert profile.inferred_frequency == "h"
