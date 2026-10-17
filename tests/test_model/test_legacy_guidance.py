# tests/test_model/test_legacy_guidance.py
"""Today's guidance blurbs fit the finding contract without losing anything.

The sample is real: blurbs from a `dsbf profile` run on the vendored titanic data,
chosen to cover every shape the survey found (a `good` level, the whole-dataset
key, an action that names another column, list and mapping metrics, EDA blurbs
with actions). The mapping below is the one the legacy adapter will implement; the
code, dimension and confidence are new facts the old blurbs never had.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from dsbf.eda.task_result import TaskResult, error_to_metadata
from dsbf.model import (
    Action,
    CheckResult,
    CheckStatus,
    Confidence,
    Dimension,
    Evidence,
    Finding,
    Guidance,
    Note,
    Severity,
    Subject,
    Timing,
)

from .builders import round_trip, strict_loads

pytestmark = pytest.mark.unit

SAMPLE = json.loads((Path(__file__).parent / "legacy_guidance_sample.json").read_text())
DATASET_KEY = "__dataset__"
ACTION_FIELDS = ("action", "method", "condition", "detail")


def task_results() -> list[TaskResult]:
    return [
        TaskResult(name=name, guidance=guidance)
        for name, guidance in SAMPLE["guidance"].items()
    ]


def legacy_blurbs(result: TaskResult):
    for column, phases in (result.guidance or {}).items():
        for phase, blurbs in phases.items():
            for index, blurb in enumerate(blurbs):
                yield column, phase, index, blurb


def to_action(legacy: dict[str, Any]) -> Action:
    columns = (
        [legacy["column"]] if "column" in legacy else list(legacy.get("columns", ()))
    )
    known = {*ACTION_FIELDS, "column", "columns"}
    return Action.model_validate(
        {
            **{k: legacy.get(k) for k in ACTION_FIELDS},
            "columns": columns,
            "params": {k: v for k, v in legacy.items() if k not in known},
        }
    )


def to_finding(result: TaskResult, column, phase, index, blurb) -> Finding:
    rows = SAMPLE["rows"]
    n = blurb["metric"].get("n")
    good = blurb["level"] == "good"
    return Finding.build(
        check_id=result.name,
        check_version="1",
        code=f"{phase}_{index}",
        subject=Subject.dataset() if column == DATASET_KEY else Subject.column(column),
        dimension=Dimension.DISTRIBUTION,
        severity=Severity.INFO if good else Severity(blurb["level"]),
        confidence=Confidence.MODERATE,
        title=blurb["title"],
        message=blurb["body"],
        evidence=Evidence(
            metrics=blurb["metric"],
            n_used=n if isinstance(n, int) and n <= rows else rows,
            n_total=rows,
            method=result.name,
        ),
        guidance=Guidance(
            phase=blurb["phase"], actions=tuple(to_action(a) for a in blurb["actions"])
        ),
        tags=("good",) if good else (),
    )


def normal_action(legacy: dict[str, Any]) -> dict[str, Any]:
    """An action in a form that ignores `column` versus `columns`."""
    action = to_action(legacy)
    return {**action.model_dump(), "columns": list(action.columns)}


def normal_blurb(blurb: dict[str, Any], column: str) -> dict[str, Any]:
    def tupled(value):
        return tuple(value) if isinstance(value, list) else value

    return {
        "column": column,
        "phase": blurb["phase"],
        "level": blurb["level"],
        "title": blurb["title"],
        "body": blurb["body"],
        "metric": {k: tupled(v) for k, v in blurb["metric"].items()},
        "actions": [normal_action(a) for a in blurb["actions"]],
    }


def blurb_from_finding(finding: Finding) -> dict[str, Any]:
    """Read a blurb back out of a finding, using only the finding's fields."""
    assert finding.guidance is not None
    return {
        "column": finding.subject.columns[0]
        if finding.subject.columns
        else DATASET_KEY,
        "phase": finding.guidance.phase,
        "level": "good" if "good" in finding.tags else finding.severity.value,
        "title": finding.title,
        "body": finding.message,
        "metric": dict(finding.evidence.metrics),
        "actions": [
            {**a.model_dump(), "columns": list(a.columns)}
            for a in finding.guidance.actions
        ],
    }


def all_findings() -> list[tuple[dict[str, Any], str, Finding]]:
    return [
        (blurb, column, to_finding(result, column, phase, index, blurb))
        for result in task_results()
        for column, phase, index, blurb in legacy_blurbs(result)
    ]


def test_the_sample_covers_every_shape_in_the_survey():
    blurbs = [(c, b) for r in task_results() for c, _, _, b in legacy_blurbs(r)]

    assert any(b["level"] == "good" for _, b in blurbs)
    assert any(c == DATASET_KEY for c, _ in blurbs)
    assert any(a.get("column", c) != c for c, b in blurbs for a in b["actions"]), (
        "an action that names another column"
    )
    assert any("columns" in a for _, b in blurbs for a in b["actions"])
    assert any(isinstance(v, list) for _, b in blurbs for v in b["metric"].values())
    assert any(isinstance(v, dict) for _, b in blurbs for v in b["metric"].values())
    assert any(b["phase"] == "eda" and b["actions"] for _, b in blurbs)
    assert any(b["phase"] == "ml" for _, b in blurbs)
    assert {b["level"] for _, b in blurbs} >= {"info", "warn", "error", "good"}


def test_every_blurb_becomes_a_finding_and_back_without_loss():
    pairs = all_findings()

    assert len(pairs) == 11
    for blurb, column, finding in pairs:
        assert blurb_from_finding(finding) == normal_blurb(blurb, column)


def test_the_findings_survive_json_and_stay_strict():
    for _, _, finding in all_findings():
        restored = round_trip(finding)
        strict_loads(finding.model_dump_json())
        assert restored == finding
        assert blurb_from_finding(restored) == blurb_from_finding(finding)


def test_ids_are_unique_within_a_check_and_stable_across_a_rebuild():
    first = [f.id for _, _, f in all_findings()]
    second = [f.id for _, _, f in all_findings()]

    assert first == second
    assert len(set(first)) == len(first)


def test_an_action_keeps_the_other_column_it_names():
    contingency = next(
        f for _, _, f in all_findings() if f.check_id == "contingency_tables"
    )

    assert contingency.subject == Subject.column("survived")
    assert contingency.guidance is not None
    assert any(
        "survived" not in action.columns for action in contingency.guidance.actions
    )


def test_a_good_blurb_keeps_its_level_through_a_tag():
    good = next(f for b, _, f in all_findings() if b["level"] == "good")

    assert good.severity == Severity.INFO
    assert good.tags == ("good",)


def test_a_skipped_task_keeps_its_message_as_a_note():
    message = SAMPLE["skipped"]["compare_with_reference_dataset"]["message"]

    result = CheckResult(
        check_id="compare_with_reference_dataset",
        check_version="1",
        status=CheckStatus.SKIPPED,
        notes=(Note(code="skipped", message=message),),
        timing=Timing(wall_seconds=0),
    )

    assert "No reference dataset provided" in result.notes[0].message
    assert round_trip(result) == result


def test_a_failed_task_keeps_its_error_metadata_as_a_note():
    metadata = error_to_metadata(ValueError("bad column [/x]"))

    note = Note(
        code="check_error",
        message="Task failed.",
        exception_type=metadata["error_type"],
        detail=metadata["trace_summary"],
        hint=metadata["suggested_action"],
    )

    assert (note.exception_type, note.detail, note.hint) == (
        "ValueError",
        "bad column [/x]",
        "Check logs or upstream outputs",
    )
    assert round_trip(note) == note
