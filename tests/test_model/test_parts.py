# tests/test_model/test_parts.py
"""The small models inside evidence and results: intervals, sampling, timing."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from dsbf.model import AssumptionCheck, Interval, SamplingInfo, Timing

pytestmark = pytest.mark.unit


def round_trip(model):
    return type(model).model_validate_json(model.model_dump_json())


class TestInterval:
    def test_round_trips(self):
        interval = Interval(low=-0.5, high=0.5, level=0.9, method="bootstrap")

        assert round_trip(interval) == interval

    def test_a_degenerate_interval_is_allowed(self):
        assert Interval(low=1.0, high=1.0, level=0.95).low == 1.0

    @pytest.mark.parametrize(
        "fields",
        [
            {"low": 2.0, "high": 1.0, "level": 0.95},
            {"low": 0.0, "high": 1.0, "level": 0.0},
            {"low": 0.0, "high": 1.0, "level": 1.0},
            {"low": float("nan"), "high": 1.0, "level": 0.95},
            {"low": 0.0, "high": float("inf"), "level": 0.95},
            {"low": 0.0, "high": 1.0, "level": float("nan")},
        ],
    )
    def test_invalid_intervals_are_rejected(self, fields):
        with pytest.raises(ValidationError):
            Interval(**fields)


class TestAssumptionCheck:
    def test_can_say_it_could_not_be_assessed(self):
        check = AssumptionCheck(name="normality")

        assert check.satisfied is None
        assert round_trip(check) == check

    def test_needs_a_name(self):
        with pytest.raises(ValidationError):
            AssumptionCheck(name="")


class TestSamplingInfo:
    def test_head_sample(self):
        info = SamplingInfo(strategy="head", original_rows=100, sampled_rows=10)

        assert info.seed is None
        assert round_trip(info) == info

    def test_random_sample_records_its_seed(self):
        info = SamplingInfo(
            strategy="random", original_rows=100, sampled_rows=10, seed=7
        )

        assert round_trip(info) == info

    def test_stratified_sample_names_its_columns(self):
        info = SamplingInfo(
            strategy="stratified",
            original_rows=100,
            sampled_rows=10,
            stratify_by=("a|b", ""),
        )

        assert round_trip(info).stratify_by == ("a|b", "")

    def test_the_legacy_fields_are_a_subset(self):
        legacy = {"original_rows": 5, "sampled_rows": 3, "strategy": "head"}

        assert SamplingInfo.model_validate(legacy).sampled_rows == 3

    @pytest.mark.parametrize(
        "fields",
        [
            {"strategy": "random"},
            {"strategy": "head", "seed": 1},
            {"strategy": "stratified"},
            {"strategy": "head", "stratify_by": ("a",)},
            {"strategy": "head", "sampled_rows": 101},
            {"strategy": "head", "original_rows": -1, "sampled_rows": 0},
            {"strategy": "bootstrap"},
        ],
    )
    def test_inconsistent_samples_are_rejected(self, fields):
        base = {"strategy": "head", "original_rows": 100, "sampled_rows": 10}

        with pytest.raises(ValidationError):
            SamplingInfo.model_validate({**base, **fields})


class TestTiming:
    def test_round_trips_with_a_timestamp(self):
        started = datetime(2026, 10, 17, 12, 30, tzinfo=UTC)
        timing = Timing(
            wall_seconds=1.25,
            cpu_seconds=1.0,
            peak_memory_bytes=2048,
            started_at=started,
        )

        restored = round_trip(timing)

        assert restored == timing
        assert restored.started_at == started

    def test_only_wall_time_is_required(self):
        timing = Timing(wall_seconds=0.0)

        assert timing.cpu_seconds is None
        assert timing.peak_memory_bytes is None
        assert timing.started_at is None

    def test_the_timestamp_is_written_as_iso_8601(self):
        started = datetime(2026, 10, 17, 12, 30, tzinfo=timezone(timedelta(hours=2)))

        dumped = Timing(wall_seconds=1, started_at=started).model_dump(mode="json")

        assert dumped["started_at"] == "2026-10-17T12:30:00+02:00"

    @pytest.mark.parametrize(
        "fields",
        [
            {"wall_seconds": -0.1},
            {"wall_seconds": float("nan")},
            {"wall_seconds": float("inf")},
            {"wall_seconds": 1.0, "cpu_seconds": -1.0},
            {"wall_seconds": 1.0, "peak_memory_bytes": -1},
            {"wall_seconds": 1.0, "started_at": datetime(2026, 10, 17, 12, 30)},
        ],
    )
    def test_invalid_timings_are_rejected(self, fields):
        with pytest.raises(ValidationError):
            Timing(**fields)

    def test_memory_is_described_as_approximate(self):
        schema = Timing.model_json_schema()

        assert "approximate" in schema["properties"]["peak_memory_bytes"]["description"]
