"""The resampling the time-series checks share, and the three analysis checks."""

from datetime import UTC, datetime, timedelta
from typing import Literal

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from dsbf.checks import core_pack
from dsbf.checks.stationarity import StationarityParams
from dsbf.checks.timeseries import parse_frequency, read_instants
from dsbf.data import load
from dsbf.engine import Registry, Selection, execute, make_plan
from dsbf.model import CheckStatus, DatetimeProfile, Severity
from dsbf.profiling import profile_dataset
from tests.test_checks.support import actions_of, output_of, run_core_check

pytestmark = pytest.mark.unit

START = datetime(2022, 1, 1)


def rng(seed: int = 3) -> np.random.Generator:
    return np.random.default_rng(seed)


def daily(values: dict[str, np.ndarray], start: datetime = START) -> pl.DataFrame:
    n = len(next(iter(values.values())))
    return pl.DataFrame({"t": [start + timedelta(days=i) for i in range(n)], **values})


def walk(n: int = 400) -> np.ndarray:
    return np.cumsum(rng().normal(size=n))


def noise(n: int = 400) -> np.ndarray:
    return rng(4).normal(size=n)


def ar1(n: int = 400, phi: float = 0.8) -> np.ndarray:
    g, x = rng(5), np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + g.normal()
    return x


def sine(n: int = 400, period: int = 7) -> np.ndarray:
    return 3 * np.sin(2 * np.pi * np.arange(n) / period) + rng(6).normal(size=n) * 0.4


class TestFrequency:
    @pytest.mark.parametrize(
        ("alias", "micros", "months"),
        [
            ("D", 86_400_000_000, None),
            ("2W", 14 * 86_400_000_000, None),
            ("15min", 900_000_000, None),
            ("h", 3_600_000_000, None),
            ("MS", None, 1),
            ("2ME", None, 2),
            ("QS", None, 3),
            ("YE", None, 12),
        ],
    )
    def test_an_alias_is_a_fixed_span_or_a_count_of_months(self, alias, micros, months):
        step = parse_frequency(alias)

        assert (step.alias, step.micros, step.months) == (alias, micros, months)

    @pytest.mark.parametrize("alias", ["", "fortnight", "0D", "D2", "3"])
    def test_anything_else_is_refused(self, alias):
        with pytest.raises(ValueError, match="offset alias|supported units"):
            parse_frequency(alias)

    def test_a_bad_frequency_is_a_parameter_error(self):
        with pytest.raises(ValidationError, match="offset alias|supported units"):
            StationarityParams(frequency="fortnight")


class TestReadInstants:
    def profile(self, **fields) -> DatetimeProfile:
        return DatetimeProfile.model_validate(
            {
                "name": "t",
                "position": 0,
                "physical_dtype": "x",
                "counts": {"rows": 4, "nulls": 0, "distinct": 4},
                **fields,
            }
        )

    def test_native_datetimes_are_microseconds_since_the_epoch(self):
        raw = np.array(["2024-01-01T00:00:01", "NaT"], dtype="datetime64[us]")

        micros, present = read_instants(raw, self.profile())

        assert present.tolist() == [True, False]
        assert micros[0] == 1_704_067_201_000_000

    def test_text_is_read_with_the_profiles_format_and_bad_values_are_absent(self):
        raw = np.array(["2024-01-01", "oops", None, " 2024-01-02 "], dtype=object)

        micros, present = read_instants(raw, self.profile(format="%Y-%m-%d"))

        assert present.tolist() == [True, False, False, True]
        assert micros[3] - micros[0] == 86_400_000_000

    def test_a_zulu_suffix_reads_as_utc(self):
        raw = np.array(["2024-01-01T00:00:00Z"], dtype=object)

        micros, present = read_instants(raw, self.profile(format="%Y-%m-%dT%H:%M:%SZ"))

        assert present.all()
        assert micros[0] == 1_704_067_200_000_000

    @pytest.mark.parametrize(("unit", "scale"), [("s", 1_000_000), ("ms", 1_000)])
    def test_unix_timestamps_are_scaled_by_their_unit(self, unit, scale):
        raw = np.array([1_700_000_000.0, np.nan, 1_700_000_060.0])

        micros, present = read_instants(raw, self.profile(epoch_unit=unit))

        assert present.tolist() == [True, False, True]
        assert micros[2] - micros[0] == 60 * scale

    def test_a_timestamp_beyond_the_arithmetic_is_absent(self):
        raw = np.array([1e25, 1_700_000_000.0])

        _, present = read_instants(raw, self.profile(epoch_unit="s"))

        assert present.tolist() == [False, True]


class TestResampling:
    def test_repeated_timestamps_are_aggregated_to_the_period(self):
        n = 300
        frame = pl.DataFrame(
            {
                "t": [START + timedelta(days=i // 3) for i in range(n)],
                "x": walk(n),
            }
        )

        result = run_core_check(frame, "stationarity", time_index="t")

        column = output_of(result)["columns"]["x"]
        assert (column["periods"], column["rows_used"]) == (100, 300)
        assert output_of(result)["periods_total"] == 100
        assert output_of(result)["aggregate"] == "mean"

    def test_the_aggregate_is_a_parameter(self):
        frame = pl.DataFrame(
            {"t": [START + timedelta(days=i // 2) for i in range(200)], "x": walk(200)}
        )

        mean = run_core_check(frame, "autocorrelation", time_index="t")
        last = run_core_check(
            frame, "autocorrelation", time_index="t", aggregate="last"
        )

        assert output_of(last)["aggregate"] == "last"
        assert (
            output_of(mean)["columns"]["x"]["acf"]
            != output_of(last)["columns"]["x"]["acf"]
        )

    def test_missing_periods_are_filled_and_counted(self):
        frame = daily({"x": walk(200)}).filter(
            ~pl.col("t").is_in([START + timedelta(days=d) for d in (20, 21, 70, 130)])
        )

        result = run_core_check(frame, "stationarity", time_index="t")

        column = output_of(result)["columns"]["x"]
        assert column["periods"] == 200
        assert column["periods_filled"] == 4
        assert column["rows_used"] == 196

    def test_a_column_with_too_many_empty_periods_has_no_series(self):
        values = walk(200)
        frame = daily({"x": values, "sparse": values.copy()}).with_columns(
            pl.when(pl.int_range(pl.len()) % 2 == 0)
            .then(None)
            .otherwise(pl.col("sparse"))
            .alias("sparse")
        )

        result = run_core_check(frame, "stationarity", time_index="t")

        output = output_of(result)
        assert "sparse" not in output["columns"]
        assert "have no value" in output["skipped"]["sparse"]
        assert "x" in output["columns"]

    def test_max_fill_share_decides_how_many_may_be_filled(self):
        values = walk(200)
        frame = daily({"x": values}).with_columns(
            pl.when(pl.int_range(pl.len()) % 4 == 1)
            .then(None)
            .otherwise(pl.col("x"))
            .alias("x")
        )

        refused = run_core_check(frame, "stationarity", time_index="t")
        allowed = run_core_check(
            frame, "stationarity", time_index="t", max_fill_share=0.4
        )

        assert refused.status is CheckStatus.NOT_APPLICABLE
        assert output_of(allowed)["columns"]["x"]["periods_filled"] > 0

    def test_only_the_most_recent_periods_are_used(self):
        result = run_core_check(
            daily({"x": walk(3000)}), "stationarity", time_index="t", max_periods=500
        )

        output = output_of(result)
        assert output["columns"]["x"]["periods"] == 500
        assert (output["periods_total"], output["truncated"]) == (3000, True)
        assert output["columns"]["x"]["series_start"] == str(
            np.datetime64(START + timedelta(days=2500), "s")
        )

    def test_the_frequency_can_be_set_when_the_profile_finds_none(self):
        g = rng()
        when = [
            START + timedelta(hours=int(h))
            for h in np.sort(g.choice(24 * 60, 700, False))
        ]
        frame = pl.DataFrame({"t": when, "x": np.cumsum(g.normal(size=700))})

        stopped = run_core_check(frame, "stationarity", time_index="t")
        resampled = run_core_check(frame, "stationarity", time_index="t", frequency="D")

        assert stopped.status is CheckStatus.NOT_APPLICABLE
        assert stopped.notes[0].code == "no_regular_frequency"
        assert "frequency" in (stopped.notes[0].hint or "")
        output = output_of(resampled)
        assert (output["frequency"], output["frequency_source"]) == ("D", "configured")
        assert output["columns"]["x"]["periods"] == 60

    def test_month_starts_are_binned_by_calendar_month(self):
        months = [datetime(2015 + m // 12, m % 12 + 1, 1) for m in range(96)]
        frame = pl.DataFrame({"t": months, "x": walk(96)})

        result = run_core_check(frame, "stationarity", time_index="t")

        output = output_of(result)
        assert output["frequency"] == "MS"
        assert output["columns"]["x"]["periods"] == 96

    def test_a_weekly_index_is_a_fixed_span(self):
        when = [START + timedelta(weeks=i) for i in range(120)]
        frame = pl.DataFrame({"t": when, "x": walk(120)})

        output = output_of(run_core_check(frame, "autocorrelation", time_index="t"))

        assert output["frequency"] == "W"
        assert output["columns"]["x"]["periods"] == 120

    def test_text_dates_and_unix_timestamps_are_read_through_the_profile(self):
        values = walk(200)
        as_text = pl.DataFrame(
            {
                "t": [
                    (START + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(200)
                ],
                "x": values,
            }
        )
        as_epoch = pl.DataFrame(
            {"t": [1_700_000_000 + 86_400 * i for i in range(200)], "x": values}
        )

        for frame in (as_text, as_epoch):
            output = output_of(run_core_check(frame, "stationarity", time_index="t"))
            assert output["frequency"] == "D"
            assert output["columns"]["x"]["periods"] == 200

    def test_a_time_zone_aware_index_is_read_in_utc(self):
        when = [
            datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(300)
        ]
        frame = pl.DataFrame({"t": when, "x": walk(300)})

        output = output_of(run_core_check(frame, "stationarity", time_index="t"))

        assert output["frequency"] == "h"
        assert output["columns"]["x"]["periods"] == 300
        assert output["timezone"] == "UTC"

    def test_unsorted_rows_are_ordered_by_the_index(self):
        frame = daily({"x": np.arange(100.0)}).sample(
            fraction=1.0, shuffle=True, seed=1
        )

        output = output_of(run_core_check(frame, "autocorrelation", time_index="t"))

        assert output["columns"]["x"]["acf"][1] > 0.95

    def test_rows_without_a_timestamp_or_a_value_are_left_out(self):
        frame = daily({"x": walk(120)}).with_columns(
            pl.when(pl.int_range(pl.len()) == 10)
            .then(None)
            .otherwise(pl.col("t"))
            .alias("t"),
        )

        output = output_of(run_core_check(frame, "stationarity", time_index="t"))

        assert output["columns"]["x"]["rows_used"] == 119
        assert output["columns"]["x"]["periods_filled"] == 1

    def test_a_column_that_is_constant_by_period_has_no_series(self):
        # Two rows a day, +a and -a: the values vary, their daily mean never does.
        pairs = np.arange(1.0, 101.0)
        frame = pl.DataFrame(
            {
                "t": [START + timedelta(days=i // 2) for i in range(200)],
                "x": walk(200),
                "flat": np.ravel(np.column_stack([pairs, -pairs])),
            }
        )

        result = run_core_check(frame, "stationarity", time_index="t")

        assert output_of(result)["skipped"] == {"flat": "constant"}

    def test_value_columns_limit_the_columns(self):
        result = run_core_check(
            daily({"x": walk(100), "y": walk(100)[::-1].copy()}),
            "stationarity",
            time_index="t",
            value_columns=["y", "missing"],
        )

        output = output_of(result)
        assert list(output["columns"]) == ["y"]
        assert "missing" in output["skipped"]


class TestSampling:
    def run(self, strategy: Literal["head", "random"], **fields):
        dataset = load(daily({"x": walk(600)}), name="frame").sample(
            300, strategy=strategy, seed=None if strategy == "head" else 1
        )
        profile = profile_dataset(dataset)
        plan = make_plan(
            Registry(core_pack().checks),
            Selection(include="autocorrelation"),
            params={"autocorrelation": {"time_index": "t", **fields}},
            profile=profile,
        )
        run = execute(plan, dataset, profile, dsbf_version="test", workers=1)
        return run.results["autocorrelation"]

    def test_a_random_sample_leaves_holes_and_the_result_says_so(self):
        result = self.run("random", frequency="D", max_fill_share=0.6)

        assert "sampled_series" in {n.code for n in result.notes}
        column = output_of(result)["columns"]["x"]
        assert column["periods_filled"] > 0
        assert column["rows_used"] == 300

    def test_a_head_sample_is_a_shorter_series_and_needs_no_note(self):
        result = self.run("head")

        assert "sampled_series" not in {n.code for n in result.notes}
        assert output_of(result)["columns"]["x"]["periods"] == 300


class TestWhenTheCheckCannotRun:
    @pytest.mark.parametrize(
        "check", ["stationarity", "autocorrelation", "seasonal_decomposition"]
    )
    def test_without_a_time_index_it_is_not_applicable_with_a_hint(self, check):
        result = run_core_check(daily({"x": walk(100)}), check)

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert result.notes[0].code == "no_time_index"
        assert "time_index" in (result.notes[0].hint or "")

    @pytest.mark.parametrize(
        "check", ["stationarity", "autocorrelation", "seasonal_decomposition"]
    )
    def test_an_index_that_is_not_a_datetime_column_is_refused(self, check):
        frame = daily({"x": walk(100), "y": walk(100)})

        result = run_core_check(frame, check, time_index="y")

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert result.notes[0].code == "time_index_not_a_datetime"

    def test_too_few_periods_is_not_applicable(self):
        result = run_core_check(daily({"x": walk(12)}), "stationarity", time_index="t")

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert result.notes[0].code == "no_series"

    def test_without_statsmodels_it_is_unavailable_with_a_hint(self, monkeypatch):
        monkeypatch.setattr("dsbf.engine.extras.find_spec", lambda name: None)

        result = run_core_check(daily({"x": walk(100)}), "stationarity", time_index="t")

        assert result.status is CheckStatus.UNAVAILABLE
        assert "statsmodels" in result.notes[0].message


class TestStationarity:
    def test_a_random_walk_drifts_and_noise_does_not(self):
        result = run_core_check(
            daily({"walk": walk(), "noise": noise()}), "stationarity", time_index="t"
        )

        columns = output_of(result)["columns"]
        assert columns["walk"]["assessment"] == "consistent_with_non_stationary"
        assert columns["noise"]["assessment"] == "consistent_with_stationary"
        (finding,) = result.findings
        assert (finding.code, finding.subject.columns) == ("non_stationary", ("walk",))
        assert finding.severity is Severity.WARN

    def test_the_evidence_states_frequency_lag_cap_and_series_length(self):
        result = run_core_check(daily({"walk": walk()}), "stationarity", time_index="t")

        metrics = result.findings[0].evidence.metrics
        assert metrics["frequency"] == "D"
        assert metrics["frequency_source"] == "profile"
        assert metrics["periods"] == 400
        assert metrics["maxlag"] == 24
        assert metrics["max_periods"] == 2000
        assert metrics["truncated"] is False
        assert result.findings[0].evidence.n_used == 400
        assert result.findings[0].evidence.method == "adf+kpss"

    def test_the_lag_cap_is_a_parameter_and_bounded_by_the_series_length(self):
        short = run_core_check(
            daily({"walk": walk(40)}), "stationarity", time_index="t"
        )
        capped = run_core_check(
            daily({"walk": walk()}), "stationarity", time_index="t", max_lag=5
        )

        assert output_of(short)["columns"]["walk"]["max_lag"] == 18
        column = output_of(capped)["columns"]["walk"]
        assert column["max_lag"] == 5
        assert column["adf"]["lags"] <= 5
        assert column["kpss"]["lags"] <= 5

    def test_each_test_is_reported_with_its_p_value_and_critical_values(self):
        output = output_of(
            run_core_check(daily({"noise": noise()}), "stationarity", time_index="t")
        )

        column = output["columns"]["noise"]
        assert column["adf"]["rejects"] is True
        assert column["adf"]["p_value"] < 0.05
        assert set(column["adf"]["critical_values"]) == {"1%", "5%", "10%"}
        assert column["kpss"]["rejects"] is False
        assert column["kpss"]["p_value_limit"] == "at_least"

    def test_a_trend_stationary_series_has_both_tests_reject_and_is_information(self):
        n = 300
        trend = ar1(n, phi=0.7) + 0.01 * np.arange(n)
        result = run_core_check(daily({"x": trend}), "stationarity", time_index="t")

        column = output_of(result)["columns"]["x"]
        assert column["assessment"] == "ambiguous_trend_stationary_or_structural_break"
        (finding,) = result.findings
        assert (finding.code, finding.severity) == (
            "ambiguous_stationarity",
            Severity.INFO,
        )

    def test_the_tips_say_to_difference_a_drifting_series(self):
        result = run_core_check(daily({"walk": walk()}), "stationarity", time_index="t")

        (action,) = actions_of(result.findings[0])
        assert (action.action, action.method) == ("transform", "difference")

    def test_a_statsmodels_warning_is_a_note_and_nothing_is_raised(self):
        result = run_core_check(
            daily({"noise": noise()}), "stationarity", time_index="t"
        )

        assert result.status is CheckStatus.OK
        assert {n.code for n in result.notes} <= {"python_warning"}

    def test_a_huge_series_costs_what_the_cap_allows(self):
        n = 200_000
        frame = pl.DataFrame(
            {
                "t": [START + timedelta(minutes=i) for i in range(n)],
                "x": np.cumsum(noise(n)),
            }
        )

        result = run_core_check(frame, "stationarity", time_index="t")

        column = output_of(result)["columns"]["x"]
        assert column["periods"] == 2000
        assert output_of(result)["periods_total"] == n
        assert result.timing.wall_seconds < 5


@pytest.mark.slow
def test_stationarity_of_a_wide_long_frame_stays_within_budget():
    """52 500 rows over 3 000 days and 40 columns: 2 000 periods each, 80 tests.

    The task took 89 s on a frame this size in the CI baseline, because it tested
    every row with the library's own lag limit.
    """
    g = rng(11)
    n = 52_500
    frame = pl.DataFrame(
        {
            "t": [START + timedelta(days=int(i * 3000 / n)) for i in range(n)],
            **{
                f"c{i}": np.cumsum(g.normal(size=n)) if i % 2 else g.normal(size=n)
                for i in range(40)
            },
        }
    )

    result = run_core_check(frame, "stationarity", time_index="t")

    output = output_of(result)
    assert len(output["columns"]) == 40
    assert (output["periods_total"], output["truncated"]) == (3000, True)
    assert {c["periods"] for c in output["columns"].values()} == {2000}
    assert result.timing.wall_seconds < 4


class TestAutocorrelation:
    def test_an_autoregressive_series_is_flagged_at_its_first_lag(self):
        result = run_core_check(
            daily({"ar": ar1(), "noise": noise()}), "autocorrelation", time_index="t"
        )

        columns = output_of(result)["columns"]
        assert columns["ar"]["strongest_lag"] == 1
        assert columns["ar"]["acf"][1] == pytest.approx(0.8, abs=0.1)
        assert 1 in columns["ar"]["flagged_pacf"]
        assert 2 not in columns["ar"]["flagged_pacf"]
        assert columns["noise"]["flagged_acf"] == []
        (finding,) = result.findings
        assert (finding.code, finding.subject.columns) == ("autocorrelated", ("ar",))
        assert finding.severity is Severity.INFO

    def test_a_weekly_cycle_peaks_at_lag_seven(self):
        result = run_core_check(
            daily({"cycle": sine()}), "autocorrelation", time_index="t"
        )

        assert output_of(result)["columns"]["cycle"]["strongest_lag"] == 7

    def test_a_lag_outside_the_band_but_below_min_abs_is_significant_not_flagged(self):
        result = run_core_check(
            daily({"ar": ar1()}), "autocorrelation", time_index="t", min_abs=0.95
        )

        column = output_of(result)["columns"]["ar"]
        assert column["significant_acf"]
        assert column["flagged_acf"] == []
        assert result.findings == ()

    def test_the_evidence_states_the_lag_cap_and_the_series(self):
        result = run_core_check(daily({"ar": ar1()}), "autocorrelation", time_index="t")

        metrics = result.findings[0].evidence.metrics
        assert (metrics["maxlag"], metrics["periods"], metrics["frequency"]) == (
            40,
            400,
            "D",
        )
        assert metrics["strongest_lag"] == 1
        assert result.findings[0].evidence.method == "acf+pacf"

    def test_the_lags_are_capped_by_the_parameter_and_the_series_length(self):
        capped = run_core_check(
            daily({"ar": ar1()}), "autocorrelation", time_index="t", max_lags=10
        )
        short = run_core_check(
            daily({"ar": ar1(30)}), "autocorrelation", time_index="t"
        )

        assert len(output_of(capped)["columns"]["ar"]["acf"]) == 11
        assert output_of(short)["columns"]["ar"]["max_lags"] == 14

    def test_a_slowly_decaying_series_is_told_to_check_stationarity(self):
        result = run_core_check(
            daily({"walk": walk()}), "autocorrelation", time_index="t"
        )

        assert "stationarity" in result.findings[0].message

    def test_the_tips_split_by_time_and_add_lags(self):
        result = run_core_check(daily({"ar": ar1()}), "autocorrelation", time_index="t")

        assert {a.action for a in actions_of(result.findings[0])} == {
            "stratified_split",
            "add_feature",
        }


class TestSeasonalDecomposition:
    def test_a_weekly_cycle_on_a_daily_index_is_found_with_period_seven(self):
        result = run_core_check(
            daily({"cycle": sine(), "walk": walk(), "noise": noise()}),
            "seasonal_decomposition",
            time_index="t",
        )

        output = output_of(result)
        assert (output["period"], output["period_source"]) == (7, "D")
        assert output["columns"]["cycle"]["seasonal_strength"] > 0.9
        found = {(f.subject.columns[0], f.code) for f in result.findings}
        assert ("cycle", "seasonal") in found
        assert ("walk", "trend") in found
        assert not {c for c, _ in found} & {"noise"}

    def test_the_output_keeps_strengths_not_component_series(self):
        output = output_of(
            run_core_check(
                daily({"cycle": sine()}), "seasonal_decomposition", time_index="t"
            )
        )

        assert set(output["columns"]["cycle"]) == {
            "periods",
            "periods_filled",
            "rows_used",
            "series_start",
            "period",
            "robust",
            "trend_strength",
            "seasonal_strength",
            "remainder_share",
        }

    def test_the_period_can_be_configured(self):
        result = run_core_check(
            daily({"cycle": sine(period=14)}),
            "seasonal_decomposition",
            time_index="t",
            period=14,
        )

        output = output_of(result)
        assert (output["period"], output["period_source"]) == (14, "configured")
        assert output["columns"]["cycle"]["seasonal_strength"] > 0.9

    def test_monthly_data_has_a_yearly_cycle(self):
        n = 96
        months = [datetime(2015 + m // 12, m % 12 + 1, 1) for m in range(n)]
        values = 5 * np.sin(2 * np.pi * np.arange(n) / 12) + noise(n) * 0.3
        frame = pl.DataFrame({"t": months, "x": values})

        output = output_of(
            run_core_check(frame, "seasonal_decomposition", time_index="t")
        )

        assert (output["frequency"], output["period"]) == ("MS", 12)

    @pytest.mark.parametrize(
        ("step", "period"),
        [
            (timedelta(hours=1), 24),
            (timedelta(hours=6), 4),
            (timedelta(minutes=15), 4),
            (timedelta(weeks=1), 52),
            (timedelta(weeks=4), 13),
            (timedelta(days=1), 7),
        ],
    )
    def test_the_period_is_one_cycle_of_the_next_longer_unit(self, step, period):
        n = 3 * period + 5
        frame = pl.DataFrame({"t": [START + i * step for i in range(n)], "x": walk(n)})

        output = output_of(
            run_core_check(frame, "seasonal_decomposition", time_index="t")
        )

        assert output["period"] == period

    def test_quarterly_data_has_a_yearly_cycle_of_four(self):
        n = 40
        quarters = [datetime(2010 + q // 4, 3 * (q % 4) + 1, 1) for q in range(n)]
        frame = pl.DataFrame({"t": quarters, "x": walk(n)})

        output = output_of(
            run_core_check(frame, "seasonal_decomposition", time_index="t")
        )

        assert (output["frequency"], output["period"]) == ("3MS", 4)

    def test_a_step_that_does_not_divide_the_cycle_implies_none(self):
        n = 40
        frame = pl.DataFrame(
            {"t": [START + timedelta(days=2 * i) for i in range(n)], "x": walk(n)}
        )

        result = run_core_check(frame, "seasonal_decomposition", time_index="t")

        assert result.notes[0].code == "no_seasonal_period"

    def test_a_frequency_without_a_natural_cycle_is_not_applicable_with_a_hint(self):
        when = [datetime(2000 + i, 1, 1) for i in range(40)]
        frame = pl.DataFrame({"t": when, "x": walk(40)})

        result = run_core_check(frame, "seasonal_decomposition", time_index="t")

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert result.notes[0].code == "no_seasonal_period"
        assert "period" in (result.notes[0].hint or "")

    def test_a_series_shorter_than_two_cycles_is_skipped_with_the_reason(self):
        result = run_core_check(
            daily({"x": walk(12)}), "seasonal_decomposition", time_index="t"
        )

        assert result.status is CheckStatus.NOT_APPLICABLE

    def test_one_series_too_short_for_the_period_does_not_stop_the_others(self):
        frame = daily({"x": sine(), "y": walk()}).with_columns(
            pl.when(pl.int_range(pl.len()) < 395)
            .then(None)
            .otherwise(pl.col("y"))
            .alias("y")
        )

        result = run_core_check(frame, "seasonal_decomposition", time_index="t")

        output = output_of(result)
        assert "x" in output["columns"]
        assert "y" in output["skipped"]

    def test_the_evidence_states_the_period_and_the_series(self):
        result = run_core_check(
            daily({"cycle": sine()}), "seasonal_decomposition", time_index="t"
        )

        finding = next(f for f in result.findings if f.code == "seasonal")
        assert finding.evidence.metrics["period"] == 7
        assert finding.evidence.metrics["periods"] == 400
        assert finding.evidence.metrics["frequency"] == "D"
        assert finding.evidence.method == "stl"
        assert {a.method for a in actions_of(finding)} == {"seasonal_difference"}
