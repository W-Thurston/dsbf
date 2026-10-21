"""Recommendation tips: declarative conditions, attached after a check returns."""

import ast
from pathlib import Path

import pytest
import yaml

from dsbf.checks import core_pack
from dsbf.engine import tips
from dsbf.model import (
    Confidence,
    Dimension,
    Evidence,
    Finding,
    Guidance,
    Severity,
    Subject,
)

pytestmark = pytest.mark.unit

SRC = Path(tips.__file__).parents[1]


def condition(text: str) -> tips.Compare | tips.AnyOf | tips.AllOf:
    return tips.Tip.model_validate(
        {"condition": yaml.safe_load(text), "action": "drop", "message": "m"}
    ).condition


def finding(metrics: dict, *, code: str = "demo_code", guidance=None) -> Finding:
    return Finding.build(
        check_id="demo",
        check_version="1.0.0",
        subject=Subject.column("a"),
        code=code,
        dimension=Dimension.USABILITY,
        severity=Severity.WARN,
        confidence=Confidence.HIGH,
        title="t",
        message="m",
        evidence=Evidence(metrics=metrics, n_used=1, n_total=1, method="demo"),
        guidance=guidance,
    )


@pytest.mark.parametrize(
    ("text", "value", "expected"),
    [
        ("{field: x, op: '<', value: 1}", 0, True),
        ("{field: x, op: '<', value: 1}", 1, False),
        ("{field: x, op: '<=', value: 1}", 1, True),
        ("{field: x, op: '>', value: 1}", 1, False),
        ("{field: x, op: '>=', value: 1}", 1, True),
        ("{field: x, op: '==', value: 1.0}", 1, True),
        ("{field: x, op: '!=', value: 1}", 2, True),
        ("{field: x, op: in, value: [a, b]}", "b", True),
        ("{field: x, op: in, value: [a, b]}", "c", False),
    ],
)
def test_every_operator(text, value, expected):
    assert tips.holds(condition(text), {"x": value}) is expected


def test_any_and_all_combine_conditions():
    either = condition(
        "{any: [{field: x, op: '>', value: 1}, {field: x, op: '<', value: -1}]}"
    )
    both = condition(
        "{all: [{field: x, op: '>', value: 1}, {field: y, op: '<', value: 5}]}"
    )

    assert tips.holds(either, {"x": -2})
    assert not tips.holds(either, {"x": 0})
    assert tips.holds(both, {"x": 2, "y": 4})
    assert not tips.holds(both, {"x": 2, "y": 9})


@pytest.mark.parametrize("values", [{}, {"x": None}, {"x": "text"}, {"x": [1]}])
def test_a_value_a_comparison_cannot_use_is_no_match(values):
    assert not tips.holds(condition("{field: x, op: '>', value: 1}"), values)


@pytest.mark.parametrize(
    "text",
    [
        "x > 1",
        "{field: x, op: '~', value: 1}",
        "{field: x, op: in, value: 1}",
        "{field: x, op: '>', value: [1, 2]}",
        "{field: x, op: '>', value: 1, extra: 2}",
        "{any: []}",
    ],
)
def test_a_condition_that_is_not_in_the_format_is_refused(text):
    with pytest.raises(ValueError):
        condition(text)


def test_the_packaged_library_is_valid_and_every_tip_names_who_it_advises():
    library = tips.load_library()

    assert library
    for name, tip in library.items():
        assert tip.checks or tip.applicable_to, name


def test_every_check_a_tip_advises_is_in_the_core_pack():
    core = {check.id for check in core_pack().checks}

    for name, tip in tips.load_library().items():
        assert set(tip.checks) <= core, name


def test_no_code_in_the_package_calls_eval():
    calls = [
        f"{path.relative_to(SRC)}:{node.lineno}"
        for path in SRC.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"eval", "exec"}
    ]

    assert calls == []


def test_a_matching_tip_becomes_an_action_on_the_finding():
    found = finding({"n_unique": 99}, code="high_cardinality")

    (tipped,), notes = tips.attach_tips("high_cardinality", [found])

    assert notes == ()
    assert tipped.guidance is not None
    assert tipped.guidance.phase == "ml"
    action = tipped.guidance.actions[0]
    assert (action.action, action.method, action.columns) == (
        "encode",
        "frequency_encoding",
        ("a",),
    )
    assert action.detail is not None
    assert "high cardinality" in action.detail


def test_a_finding_below_the_condition_gets_no_tip():
    found = finding({"n_unique": 3})

    assert tips.attach_tips("high_cardinality", [found]) == ((found,), ())


def test_a_check_with_no_tips_keeps_its_findings():
    found = finding({"n_unique": 99})

    assert tips.attach_tips("no_such_check", [found]) == ((found,), ())


def test_a_finding_that_has_guidance_keeps_it():
    own = Guidance(phase="eda", actions=())
    found = finding({"n_unique": 99}, guidance=own)

    (kept,), _ = tips.attach_tips("high_cardinality", [found])

    assert kept.guidance is own


def test_tips_are_limited_to_their_finding_codes():
    zero = finding({"zero_share": 0.6}, code="high_zero_rate")
    other = finding({"zero_share": 0.6}, code="something_else")

    (tipped, untouched), _ = tips.attach_tips("zeros", [zero, other])

    assert tipped.guidance is not None
    assert [a.method for a in tipped.guidance.actions] == ["log1p", "is_zero", None]
    assert untouched.guidance is None


def test_an_unusable_library_is_a_note_and_the_findings_are_untouched(monkeypatch):
    def broken():
        raise tips.TipLibraryError("gone")

    monkeypatch.setattr(tips, "load_library", broken)
    found = finding({"n_unique": 99})

    findings, notes = tips.attach_tips("high_cardinality", [found])

    assert findings == (found,)
    assert [n.code for n in notes] == ["tip_library_unavailable"]


def test_a_lookup_that_raises_is_a_counted_note(monkeypatch):
    def explode(found, library):
        raise RuntimeError("boom")

    monkeypatch.setattr(tips, "_with_tips", explode)
    found = finding({"n_unique": 99})

    findings, notes = tips.attach_tips("high_cardinality", [found, found])

    assert findings == (found, found)
    assert [(n.code, n.count) for n in notes] == [("tip_lookup_failed", 2)]


def test_the_library_digest_follows_the_library(monkeypatch, tmp_path):
    before = tips.library_digest()
    path = tmp_path / "library.yaml"
    path.write_text(
        "only:\n  condition: {field: x, op: '>', value: 1}\n"
        "  checks: [demo]\n  action: drop\n  message: m\n",
        encoding="utf-8",
    )
    tips.load_library.cache_clear()
    monkeypatch.setattr(tips, "_resource", lambda: path)
    try:
        assert tips.library_digest() != before
    finally:
        tips.load_library.cache_clear()
