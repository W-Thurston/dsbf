# tests/test_model/test_check_result.py
"""A check result always says how it ended, and why when that was not ok."""

import pandas as pd
import pytest
from pydantic import ValidationError

from dsbf.model import CheckResult, CheckStatus, Severity, Timing

from .builders import (
    make_finding,
    make_note,
    make_result,
    round_trip,
    strict_loads,
)

pytestmark = pytest.mark.unit

NOT_OK = [status for status in CheckStatus if status != CheckStatus.OK]


def test_the_statuses_are_the_six_in_the_design():
    assert [s.value for s in CheckStatus] == [
        "ok",
        "not_applicable",
        "unavailable",
        "skipped",
        "error",
        "timeout",
    ]


def test_an_ok_result_round_trips():
    result = make_result(
        output={"rate": 0.4, "by_column": {"age": 0.4}}, params={"threshold": 0.3}
    )

    assert round_trip(result) == result


@pytest.mark.parametrize("status", NOT_OK)
def test_every_other_status_round_trips_with_its_note(status):
    result = make_result(status)

    assert round_trip(result) == result
    assert result.notes[0].message


@pytest.mark.parametrize("status", NOT_OK)
def test_a_status_other_than_ok_needs_a_note(status):
    with pytest.raises(ValidationError, match="note"):
        CheckResult(
            check_id="c",
            check_version="1",
            status=status,
            timing=Timing(wall_seconds=0),
        )


@pytest.mark.parametrize("status", NOT_OK)
def test_a_status_other_than_ok_has_no_findings(status):
    with pytest.raises(ValidationError, match="no findings"):
        make_result(status, findings=(make_finding(),))


def test_an_ok_result_may_have_no_findings_and_no_notes():
    result = make_result(findings=())

    assert result.findings == () and result.notes == ()


@pytest.mark.parametrize("mismatch", [{"check_id": "other"}, {"check_version": "2"}])
def test_a_finding_from_another_check_is_rejected(mismatch):
    with pytest.raises(ValidationError, match="belongs to"):
        make_result(findings=(make_finding(**mismatch),))


def test_an_unknown_status_is_rejected():
    data = make_result().model_dump(mode="json")
    data["status"] = "success"

    with pytest.raises(ValidationError):
        CheckResult.model_validate(data)


class TestOutput:
    def test_undefined_numbers_become_null_and_name_their_key(self):
        result = make_result(
            output={
                "r": float("nan"),
                "when": pd.NaT,
                "n": 3,
                "rows": [1.0, float("inf")],
            }
        )

        assert result.output == {"r": None, "when": None, "n": 3, "rows": [1.0, None]}
        assert result.non_finite_outputs == ("r", "rows", "when")
        strict_loads(result.model_dump_json())
        assert round_trip(result) == result

    def test_a_clean_output_flags_nothing(self):
        assert make_result(output={"n": 1}).non_finite_outputs == ()

    def test_values_that_are_not_json_are_rejected(self):
        with pytest.raises(ValidationError):
            make_result(output={"frame": object()})

    def test_keys_must_be_strings(self):
        with pytest.raises(ValidationError):
            make_result(output={1: "a"})


class TestParams:
    def test_are_recorded(self):
        assert make_result(params={"k": 3, "name": "x"}).params == {"k": 3, "name": "x"}

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_must_be_finite(self, value):
        with pytest.raises(ValidationError, match="finite"):
            make_result(params={"limit": value})


class TestNote:
    def test_carries_a_skip_reason_and_its_blocker(self):
        note = make_note(blocked_by=("skewness",), severity=Severity.WARN)

        assert round_trip(note) == note

    def test_carries_an_error(self):
        note = make_note(
            code="check_error",
            message="Comparison budget exceeded.",
            hint="Lower max_rows.",
            exception_type="TimeoutError",
            detail="after 300s",
        )

        assert round_trip(note) == note

    def test_counts_a_repeated_event(self):
        note = make_note(code="python_warning", count=7)

        assert round_trip(note).count == 7
        assert make_note().count is None

    @pytest.mark.parametrize("count", [0, -1])
    def test_a_count_is_at_least_one(self, count):
        with pytest.raises(ValidationError):
            make_note(count=count)

    @pytest.mark.parametrize("field", ["code", "message"])
    def test_needs_a_code_and_a_message(self, field):
        with pytest.raises(ValidationError):
            make_note(**{field: ""})
