"""The datetime rule checks on small frames: consistency, gaps and features."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from dsbf.model import CheckStatus, Severity
from tests.test_checks.support import actions_of, output_of, run_core_check

pytestmark = pytest.mark.unit


def days(count: int, start: datetime = datetime(2024, 1, 1), step: int = 1):
    return [start + timedelta(days=step * i) for i in range(count)]


class TestDatetimeConsistency:
    def text_dates(self, bad: int, total: int = 100) -> pl.DataFrame:
        values: list[str | None] = [d.strftime("%Y-%m-%d") for d in days(total)]
        for i in range(bad):
            values[i * 3] = f"not a date {i}"
        return pl.DataFrame({"when": values})

    def test_values_outside_the_inferred_format_are_reported(self):
        result = run_core_check(self.text_dates(bad=8), "datetime_consistency")

        assert result.status is CheckStatus.OK
        (finding,) = result.findings
        assert finding.code == "unparsed_values"
        assert finding.severity is Severity.WARN
        assert finding.evidence.metrics["unparsed"] == 8
        assert finding.evidence.metrics["format"] == "%Y-%m-%d"
        assert finding.evidence.n_used == 100
        assert output_of(result)["columns"]["when"]["share"] == pytest.approx(0.08)

    def test_a_few_bad_values_are_information_below_the_warn_share(self):
        result = run_core_check(self.text_dates(bad=2), "datetime_consistency")

        (finding,) = result.findings
        assert finding.severity is Severity.INFO

    def test_the_warn_share_is_a_parameter(self):
        result = run_core_check(
            self.text_dates(bad=2), "datetime_consistency", warn_share=0.01
        )

        assert result.findings[0].severity is Severity.WARN

    def test_a_datetime_dtype_has_nothing_unparsed(self):
        result = run_core_check(
            pl.DataFrame({"when": days(50)}), "datetime_consistency"
        )

        assert result.findings == ()
        assert output_of(result)["columns"]["when"]["unparsed"] == 0

    def test_a_missing_value_is_not_an_unparsed_one(self):
        values: list[str | None] = [d.strftime("%Y-%m-%d") for d in days(60)]
        values[5] = None
        result = run_core_check(pl.DataFrame({"when": values}), "datetime_consistency")

        assert result.findings == ()
        assert output_of(result)["columns"]["when"]["non_null"] == 59

    def test_data_without_a_datetime_column_is_not_applicable(self):
        result = run_core_check(
            pl.DataFrame({"x": [1.5, 2.5, 3.5]}), "datetime_consistency"
        )

        assert result.status is CheckStatus.NOT_APPLICABLE


class TestTemporalGaps:
    def test_a_regular_series_has_no_finding(self):
        result = run_core_check(pl.DataFrame({"when": days(60)}), "temporal_gaps")

        assert result.findings == ()
        column = output_of(result)["columns"]["when"]
        assert (column["frequency"], column["n_gaps"]) == ("D", 0)

    def test_missing_days_are_gaps(self):
        kept = [d for i, d in enumerate(days(60)) if i not in (10, 11, 30)]
        result = run_core_check(pl.DataFrame({"when": kept}), "temporal_gaps")

        (finding,) = result.findings
        assert finding.code == "gaps"
        assert finding.severity is Severity.INFO
        assert finding.evidence.metrics["n_gaps"] == 2
        assert finding.evidence.metrics["largest_gap_days"] == 3.0
        assert finding.evidence.metrics["frequency"] == "D"
        assert finding.evidence.method == "dominant_step_gaps"

    def test_many_gaps_are_a_warning(self):
        kept = [d for i, d in enumerate(days(80)) if i % 8 not in (3, 4)]
        result = run_core_check(pl.DataFrame({"when": kept}), "temporal_gaps")

        assert result.findings[0].severity is Severity.WARN

    def test_irregular_spacing_has_no_frequency_and_says_so(self):
        offsets = [0, 1, 3, 7, 9, 14, 17, 25, 30, 41, 44, 50, 60, 80, 99]
        when = [datetime(2024, 1, 1) + timedelta(days=o) for o in offsets]
        result = run_core_check(pl.DataFrame({"when": when}), "temporal_gaps")

        (finding,) = result.findings
        assert finding.code == "irregular_spacing"
        assert output_of(result)["columns"]["when"]["frequency"] is None

    def test_a_column_with_few_distinct_values_is_skipped(self):
        result = run_core_check(
            pl.DataFrame({"when": days(5), "other": days(60)[:5]}), "temporal_gaps"
        )

        assert result.status is CheckStatus.NOT_APPLICABLE

    def test_each_column_is_judged_on_its_own(self):
        frame = pl.DataFrame({"a": days(40), "b": days(40, step=7)})
        result = run_core_check(frame, "temporal_gaps")

        columns = output_of(result)["columns"]
        assert columns["a"]["frequency"] == "D"
        assert columns["b"]["frequency"] == "W"

    def test_gap_tips_advise_resampling_and_a_look_at_where(self):
        kept = [d for i, d in enumerate(days(60)) if i not in (10, 11, 30)]
        result = run_core_check(pl.DataFrame({"when": kept}), "temporal_gaps")

        assert {a.action for a in actions_of(result.findings[0])} == {
            "resample",
            "investigate",
        }


class TestDatetimeFeatures:
    def features(self, frame: pl.DataFrame) -> dict[str, tuple[str, ...]]:
        columns = output_of(run_core_check(frame, "datetime_features"))["columns"]
        return {name: tuple(c["features"]) for name, c in columns.items()}

    def test_a_year_of_dates_lists_calendar_features_but_no_hour_or_year(self):
        listed = self.features(pl.DataFrame({"when": days(365)}))["when"]

        assert "month" in listed
        assert "day_of_week" in listed
        assert "month_sin_cos" in listed
        assert "hour" not in listed
        assert "year" not in listed

    def test_several_years_add_the_year(self):
        listed = self.features(pl.DataFrame({"when": days(40, step=60)}))["when"]

        assert "year" in listed

    def test_a_time_of_day_adds_the_hour_features(self):
        when = [datetime(2024, 1, 1) + timedelta(hours=5 * i) for i in range(100)]
        listed = self.features(pl.DataFrame({"when": when}))["when"]

        assert "hour" in listed
        assert "hour_sin_cos" in listed

    def test_one_month_has_no_month_features_and_no_day_of_month(self):
        listed = self.features(pl.DataFrame({"when": days(20)}))["when"]

        assert "month" not in listed
        assert "day_of_month" not in listed
        assert "day_of_week" in listed

    def test_the_finding_names_the_features_and_carries_tips(self):
        result = run_core_check(pl.DataFrame({"when": days(365)}), "datetime_features")

        (finding,) = result.findings
        assert finding.code == "extractable_features"
        assert finding.severity is Severity.INFO
        assert finding.evidence.metrics["n_cyclic"] == 2
        features = finding.evidence.metrics["features"]
        assert isinstance(features, tuple)
        assert set(features) == set(output_of(result)["columns"]["when"]["features"])
        assert {a.action for a in actions_of(finding)} == {"add_feature", "encode"}

    def test_a_single_value_has_nothing_to_extract(self):
        result = run_core_check(
            pl.DataFrame({"when": [datetime(2024, 1, 1)] * 5}), "datetime_features"
        )

        assert result.status is CheckStatus.NOT_APPLICABLE
