"""Guidance blurbs become findings that keep what the blurb said."""

import warnings
from pathlib import Path
from typing import Any

import pytest

from dsbf.data import load
from dsbf.engine import CheckContext
from dsbf.legacy import LegacySession, legacy_checks
from dsbf.legacy.findings import findings_from
from dsbf.model import Confidence, Run, Severity, Subject
from dsbf.profile import profile_dataset
from tests.test_legacy.support import make_session, output_of, run_adapted
from tests.test_model.builders import round_trip, strict_loads
from tests.test_model.test_legacy_guidance import (
    DATASET_KEY,
    blurb_from_finding,
    legacy_blurbs,
    normal_blurb,
    task_results,
)

pytestmark = pytest.mark.unit

TITANIC = Path(__file__).parents[1] / "golden" / "data" / "titanic.csv"


def blurb(title: str = "Heavy Right Skew (+4.78)", **changes: Any) -> dict[str, Any]:
    return {
        "phase": "eda",
        "column": "age",
        "level": "warn",
        "title": title,
        "body": "A body.",
        "actions": [],
        "metric": {"skewness": 4.78, "n": 200},
        **changes,
    }


def guidance(*blurbs: dict[str, Any], column: str = "age") -> dict[str, Any]:
    out: dict[str, Any] = {column: {"eda": [], "ml": []}}
    for entry in blurbs:
        out[column][entry["phase"]].append(entry)
    return out


@pytest.fixture(scope="module")
def session(tmp_path_factory) -> LegacySession:
    return make_session(tmp_path_factory.mktemp("figures"))


def context(session: LegacySession, task: str) -> CheckContext:
    check = next(c for c in legacy_checks(session) if c.id == task)
    return CheckContext(check.spec, session.dataset, session.profile)


def record(session: LegacySession, task: str, **by_column: Any):
    ctx = context(session, task)
    merged: dict[str, Any] = {}
    for column, blurbs in by_column.items():
        merged.update(guidance(*blurbs, column=column))
    findings_from(ctx, merged)
    return ctx


def test_a_blurb_becomes_a_finding_with_the_blurbs_words_and_numbers(session):
    ctx = record(session, "detect_skewness", age=[blurb()])

    (finding,) = ctx.findings
    assert finding.subject == Subject.column("age")
    assert finding.title == "Heavy Right Skew (+4.78)"
    assert finding.message == "A body."
    assert finding.severity is Severity.WARN
    assert finding.confidence is Confidence.MODERATE
    assert finding.evidence.metrics == {"skewness": 4.78, "n": 200}
    assert finding.evidence.method == "detect_skewness"
    assert finding.dimension in ctx.spec.dimensions


def test_the_code_names_the_phase_and_the_words_not_the_numbers(session):
    one = record(session, "detect_skewness", age=[blurb("Heavy Right Skew (+4.78)")])
    two = record(session, "detect_skewness", age=[blurb("Heavy Right Skew (+9.10)")])
    ml = record(session, "detect_skewness", age=[blurb(phase="ml")])

    assert one.findings[0].code == "eda_heavy_right_skew"
    assert two.findings[0].id == one.findings[0].id
    assert ml.findings[0].code == "ml_heavy_right_skew"
    assert ml.findings[0].id != one.findings[0].id


def test_the_ids_do_not_depend_on_the_order_of_the_blurbs(session):
    first = blurb("Heavy Right Skew (+4.78)")
    second = blurb("Some Missingness (19.9%)")

    forward = record(session, "detect_skewness", age=[first, second])
    backward = record(session, "detect_skewness", age=[second, first])

    assert {f.id for f in forward.findings} == {f.id for f in backward.findings}


def test_a_repeated_code_for_one_subject_is_numbered_in_blurb_order(session):
    ctx = record(
        session,
        "detect_skewness",
        age=[blurb("Spike at 3"), blurb("Spike at 7"), blurb("Spike at 9")],
    )

    assert [f.code for f in ctx.findings] == [
        "eda_spike_at",
        "eda_spike_at_2",
        "eda_spike_at_3",
    ]


def test_a_long_title_is_cut_and_keeps_a_digest_of_the_whole(session):
    long = "Association with " + "a very long column name " * 6
    other = long + "differs at the end"

    codes = [
        record(session, "detect_skewness", age=[blurb(t)]).findings[0].code
        for t in (long, other)
    ]

    assert all(len(code) < 80 for code in codes)
    assert codes[0] != codes[1]


def test_a_good_blurb_is_info_with_the_good_tag(session):
    ctx = record(session, "detect_skewness", age=[blurb(level="good")])

    assert ctx.findings[0].severity is Severity.INFO
    assert ctx.findings[0].tags == ("good",)


def test_an_unknown_level_is_info_and_says_so(session):
    ctx = record(session, "detect_skewness", age=[blurb(level="critical")])

    assert ctx.findings[0].severity is Severity.INFO
    assert ctx.findings[0].tags == ("legacy_level_critical",)


def test_the_dataset_key_is_the_dataset_subject(session):
    ctx = record(session, "detect_constant_columns", **{DATASET_KEY: [blurb()]})

    assert ctx.findings[0].subject == Subject.dataset()


@pytest.mark.parametrize(
    ("metric", "used"),
    [
        ({"n": 120}, 120),
        ({}, 240),
        ({"n": 5000}, 240),
        ({"n": -1}, 240),
        ({"n": True}, 240),
        ({"n": 12.5}, 240),
    ],
)
def test_rows_used_is_the_reported_n_within_the_rows_else_all_of_them(
    session, metric, used
):
    ctx = record(session, "detect_skewness", age=[blurb(metric=metric)])

    assert ctx.findings[0].evidence.n_total == 240
    assert ctx.findings[0].evidence.n_used == used


def test_a_blurb_outside_the_checks_columns_is_left_out_and_noted(session):
    ctx = record(
        session,
        "detect_skewness",
        age=[blurb()],
        group=[blurb(column="group"), blurb("Another (3)", column="group")],
    )

    assert [f.subject.columns for f in ctx.findings] == [("age",)]
    (note,) = ctx.notes
    assert note.code == "findings_out_of_scope"
    assert "2 finding(s)" in note.message
    assert note.detail == "group"


def test_nothing_is_noted_when_every_blurb_is_in_scope(session):
    assert record(session, "detect_skewness", age=[blurb()]).notes == ()


def test_an_action_keeps_its_columns_method_and_extra_fields(session):
    actions = [
        {"action": "transform", "method": "log1p", "column": "age", "feature": "f"},
        {"action": "create_interaction", "columns": ["age", "fare"], "detail": "d"},
        {"action": "review"},
    ]

    ctx = record(session, "detect_skewness", age=[blurb(phase="ml", actions=actions)])

    stored = ctx.findings[0].guidance
    assert stored is not None
    first, second, third = stored.actions
    assert (first.columns, first.method, first.params) == (
        ("age",),
        "log1p",
        {"feature": "f"},
    )
    assert (second.columns, second.detail) == (("age", "fare"), "d")
    assert third.columns == ()
    assert stored.phase == "ml"


def test_a_sampled_dataset_stamps_its_findings_with_the_sample(session):
    sampled = session.dataset.sample(100, strategy="head")
    profile = profile_dataset(sampled)
    check = next(c for c in legacy_checks(session) if c.id == "detect_skewness")
    ctx = CheckContext(check.spec, sampled, profile)
    findings_from(ctx, guidance(blurb()))

    assert ctx.findings[0].evidence.sampled == sampled.sampling
    assert ctx.findings[0].evidence.n_total == 100


def test_the_titanic_sample_converts_without_loss():
    dataset = load(TITANIC)
    titanic = LegacySession(dataset, profile_dataset(dataset))
    for result in task_results():
        ctx = context(titanic, "detect_constant_columns")
        findings_from(ctx, result.guidance)
        expected = [
            normal_blurb(b, column) for column, _, _, b in legacy_blurbs(result)
        ]
        assert [blurb_from_finding(f) for f in ctx.findings] == expected
        for finding in ctx.findings:
            assert round_trip(finding) == finding
            strict_loads(finding.model_dump_json())


class TestThroughTheEngine:
    NAMES = ("detect_outliers", "detect_skewness", "summarize_nulls")

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, session) -> Run:
        warnings.simplefilter("ignore")
        return run_adapted(session, *cls.NAMES)

    def test_the_findings_are_in_the_result_and_the_output_keeps_the_guidance(
        self, run
    ):
        for name in self.NAMES:
            result = run.results[name]
            guidance_blurbs = sum(
                len(blurbs)
                for phases in (
                    output_of(run, name)["result"]["guidance"] or {}
                ).values()
                for blurbs in phases.values()
            )
            assert result.findings, name
            assert len(result.findings) + _left_out(result) == guidance_blurbs

    def test_finding_ids_are_unique_in_the_run(self, run):
        ids = [f.id for r in run.results.values() for f in r.findings]

        assert len(ids) == len(set(ids)) > 0

    def test_the_run_is_strict_json(self, run):
        strict_loads(run.model_dump_json())


def _left_out(result) -> int:
    for note in result.notes:
        if note.code == "findings_out_of_scope":
            return int(note.message.split()[2])
    return 0
