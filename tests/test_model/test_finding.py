# tests/test_model/test_finding.py
"""A finding has a stable id, carries its evidence and guidance, and round-trips."""

import pytest
from pydantic import ValidationError

from dsbf.model import Action, Finding, Guidance, Subject

from .builders import make_evidence, make_finding, round_trip, strict_loads

pytestmark = pytest.mark.unit


class TestId:
    def test_is_derived_by_build_and_stable(self):
        first = make_finding()
        second = make_finding(title="Another title", message="Other words")

        assert first.id == second.id
        assert len(first.id) == 16

    @pytest.mark.parametrize(
        "change",
        [
            {"check_id": "summarize_numeric"},
            {"code": "all_null"},
            {"subject": Subject.column("fare")},
            {"subject": Subject.dataset()},
        ],
    )
    def test_changes_with_check_subject_or_code(self, change):
        assert make_finding(**change).id != make_finding().id

    def test_does_not_change_with_the_check_version(self):
        assert make_finding(check_version="2").id == make_finding().id

    def test_tells_a_column_from_a_group_or_row_set_of_the_same_name(self):
        ids = {
            make_finding(subject=Subject.column_group("a", "b")).id,
            make_finding(subject=Subject.row_set("a", "b")).id,
            make_finding(subject=Subject.column_pair("a", "b")).id,
        }

        assert len(ids) == 3

    def test_keeps_the_order_of_a_pairs_columns(self):
        forward = make_finding(subject=Subject.column_pair("a", "b"))
        backward = make_finding(subject=Subject.column_pair("b", "a"))

        assert forward.id != backward.id

    def test_cannot_collide_through_a_delimiter_in_a_column_name(self):
        joined = make_finding(subject=Subject.column_pair("a|b", "c"))
        other = make_finding(subject=Subject.column_pair("a", "b|c"))

        assert joined.id != other.id

    @pytest.mark.parametrize("name", ["", "[/x]", "日本語", 'quote"d', "line\nbreak"])
    def test_accepts_awkward_column_names(self, name):
        finding = make_finding(subject=Subject.column(name))

        assert round_trip(finding) == finding

    def test_a_wrong_id_is_rejected(self):
        data = make_finding().model_dump(mode="json")
        data["id"] = "0" * 16

        with pytest.raises(ValidationError, match="does not match"):
            Finding.model_validate(data)

    def test_is_required_in_the_schema(self):
        assert "id" in Finding.model_json_schema()["required"]

    def test_is_not_derived_for_you_when_missing(self):
        data = make_finding().model_dump(mode="json", exclude={"id"})

        with pytest.raises(ValidationError, match="id"):
            Finding.model_validate(data)

    def test_make_id_matches_the_built_finding(self):
        finding = make_finding()

        assert finding.id == Finding.make_id(
            finding.check_id, finding.subject, finding.code
        )


class TestFinding:
    def test_round_trips_with_evidence_and_guidance(self):
        finding = make_finding(
            evidence=make_evidence(metrics={"null_rate": 0.4, "since": "2020"}),
            guidance=Guidance(
                phase="ml",
                actions=(
                    Action(action="impute", columns=("age",), method="median"),
                    Action(action="add_indicator", columns=("age",)),
                ),
            ),
            tags=("completeness", "good"),
        )

        assert round_trip(finding) == finding

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            make_finding(colour="red")

    def test_an_undefined_metric_still_produces_strict_json(self):
        finding = make_finding(evidence=make_evidence(metrics={"r": float("nan")}))

        strict_loads(finding.model_dump_json())
        assert finding.evidence.non_finite_metrics == ("r",)

    @pytest.mark.parametrize("field", ["check_id", "check_version", "code"])
    def test_names_cannot_be_empty(self, field):
        with pytest.raises(ValidationError):
            make_finding(**{field: ""})


class TestAction:
    def test_keeps_extra_parameters(self):
        action = Action(
            action="change_dtype",
            columns=("a",),
            detail="int64 to bool",
            params={"from_dtype": "int64", "to_dtype": "bool"},
        )

        assert round_trip(action) == action

    def test_parameters_must_be_finite(self):
        with pytest.raises(ValidationError):
            Action(action="clip", params={"upper": float("inf")})

    def test_needs_a_verb(self):
        with pytest.raises(ValidationError):
            Action(action="")


class TestGuidance:
    def test_phase_is_eda_or_ml(self):
        assert Guidance(phase="eda").actions == ()
        with pytest.raises(ValidationError):
            Guidance.model_validate({"phase": "diagnostic"})
