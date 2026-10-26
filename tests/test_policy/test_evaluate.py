"""Policy rules over hand-built runs: each matcher key, each action, no previous run."""

import pytest

from dsbf.model import (
    CheckStatus,
    Confidence,
    Dimension,
    Policy,
    Severity,
    Subject,
)
from dsbf.policy import evaluate_policy, load_policy
from tests.test_model.builders import (
    make_evidence,
    make_finding,
    make_result,
    make_run,
)

pytestmark = pytest.mark.unit


def finding(code="high_null_rate", check_id="null_rate", column="age", **overrides):
    return make_finding(
        code=code, check_id=check_id, subject=Subject.column(column), **overrides
    )


def run_of(*findings, extra=()):
    by_check: dict[str, list] = {}
    for item in findings:
        by_check.setdefault(item.check_id, []).append(item)
    results = [
        make_result(check_id=check_id, findings=tuple(items))
        for check_id, items in by_check.items()
    ]
    return make_run([*results, *extra])


def matched(run, when, previous=None, then="annotate"):
    policy = load_policy({"rules": [{"name": "r", "when": when, "then": then}]})
    outcome, _ = evaluate_policy(run, policy, previous)
    return outcome.rules[0].finding_ids


@pytest.fixture
def nulls():
    return finding(
        evidence=make_evidence(metrics={"null_share": 0.4, "flag": True, "kind": "a"})
    )


def test_each_finding_key_selects_on_its_own(nulls):
    other = finding(
        code="leaky",
        check_id="target_leakage",
        column="y",
        dimension=Dimension.LEAKAGE,
        severity=Severity.ERROR,
        confidence=Confidence.LOW,
        tags=("ml",),
    )
    run = run_of(nulls, other)
    assert matched(run, {"dimension": "leakage"}) == (other.id,)
    assert matched(run, {"code": "high_null_rate"}) == (nulls.id,)
    assert matched(run, {"check": "target_leakage"}) == (other.id,)
    assert matched(run, {"column": "age"}) == (nulls.id,)
    assert matched(run, {"tag": "ml"}) == (other.id,)
    assert matched(run, {"severity": ">=error"}) == (other.id,)
    assert matched(run, {"severity": "warn"}) == (nulls.id,)
    assert matched(run, {"confidence": "<moderate"}) == (other.id,)
    assert matched(run, {"confidence": "!=low"}) == (nulls.id,)


def test_every_key_given_must_hold(nulls):
    run = run_of(nulls)
    assert matched(run, {"code": "high_null_rate", "column": "age"}) == (nulls.id,)
    assert matched(run, {"code": "high_null_rate", "column": "fare"}) == ()


def test_a_column_matches_a_pair_on_either_side_and_never_the_dataset():
    pair = make_finding(
        check_id="associations", subject=Subject.column_pair("a", "b"), code="p"
    )
    whole = make_finding(check_id="duplicate_rows", subject=Subject.dataset(), code="d")
    run = run_of(pair, whole)
    assert matched(run, {"column": "a"}) == (pair.id,)
    assert matched(run, {"column": "b"}) == (pair.id,)
    assert matched(run, {"column": "c"}) == ()


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (">0.3", True),
        (">0.4", False),
        (">=0.4", True),
        ("<0.5", True),
        ("<=0.39", False),
        ("0.4", True),
        ("!=0.4", False),
    ],
)
def test_a_metric_compares_numbers(nulls, condition, expected):
    found = matched(run_of(nulls), {"metric": {"null_share": condition}})
    assert found == ((nulls.id,) if expected else ())


def test_text_and_flags_compare_by_equality_only(nulls):
    run = run_of(nulls)
    assert matched(run, {"metric": {"flag": "true"}}) == (nulls.id,)
    assert matched(run, {"metric": {"flag": "false"}}) == ()
    assert matched(run, {"metric": {"kind": "a"}}) == (nulls.id,)
    assert matched(run, {"metric": {"kind": "!=b"}}) == (nulls.id,)
    # A flag is not the number 1, and text is not a number.
    assert matched(run, {"metric": {"flag": "1"}}) == ()
    assert matched(run, {"metric": {"null_share": "!=a"}}) == (nulls.id,)
    assert matched(run, {"metric": {"kind": "0.4"}}) == ()


def test_a_missing_null_or_listed_metric_never_holds():
    odd = finding(
        evidence=make_evidence(
            metrics={"nan": float("nan"), "listed": (1, 2), "mapped": {"a": 1}}
        )
    )
    run = run_of(odd)
    for name in ("absent", "nan", "listed", "mapped"):
        assert matched(run, {"metric": {name: "!=5"}}) == ()
        assert matched(run, {"metric": {name: ">=0"}}) == ()


def test_every_metric_entry_must_hold(nulls):
    run = run_of(nulls)
    both = {"null_share": ">0.3", "kind": "a"}
    assert matched(run, {"metric": both}) == (nulls.id,)
    assert matched(run, {"metric": {**both, "kind": "b"}}) == ()


def test_health_rules_select_checks_by_status_and_may_be_narrowed():
    broken = make_result(CheckStatus.ERROR, check_id="stationarity")
    also = make_result(CheckStatus.ERROR, check_id="bimodality")
    run = make_run([make_result(), broken, also])
    policy = load_policy(
        {
            "rules": [
                {"name": "all", "when": {"health.status": "error"}},
                {
                    "name": "one",
                    "when": {"health.status": "error", "check": "bimodality"},
                },
                {"name": "none", "when": {"health.status": "timeout"}},
            ]
        }
    )
    outcome, _ = evaluate_policy(run, policy)
    assert [o.check_ids for o in outcome.rules] == [
        ("stationarity", "bimodality"),
        ("bimodality",),
        (),
    ]
    assert all(not o.finding_ids for o in outcome.rules)


def test_new_since_compares_ids_with_the_previous_run(nulls):
    fresh = finding(code="all_null", column="z")
    run = run_of(nulls, fresh)
    previous = run_of(nulls)
    assert matched(run, {"new_since": "previous_run"}, previous) == (fresh.id,)
    assert matched(run, {"new_since": "previous_run", "column": "age"}, previous) == ()


def test_new_since_without_a_previous_run_matches_nothing_and_says_so(nulls):
    policy = load_policy(
        {
            "rules": [
                {"name": "new", "when": {"new_since": "previous_run"}, "then": "fail"},
                {
                    "name": "also-new",
                    "when": {"code": "x", "new_since": "previous_run"},
                },
                {"name": "plain", "when": {"code": "high_null_rate"}},
            ]
        }
    )
    outcome, notes = evaluate_policy(run_of(nulls), policy)
    assert [o.triggered for o in outcome.rules] == [False, False, True]
    assert not outcome.failed
    assert [note.code for note in notes] == ["policy_no_previous_run"]
    assert "new, also-new" in notes[0].message


def test_no_note_without_new_since_or_when_a_previous_run_is_given(nulls):
    policy = load_policy({"rules": [{"name": "r", "when": {"code": "x"}}]})
    assert evaluate_policy(run_of(nulls), policy)[1] == ()
    new = load_policy({"rules": [{"name": "r", "when": {"new_since": "previous_run"}}]})
    assert evaluate_policy(run_of(nulls), new, run_of(nulls))[1] == ()


def test_every_rule_has_an_outcome_in_policy_order_fired_or_not(nulls):
    policy = load_policy(
        {
            "rules": [
                {"name": "b", "when": {"code": "nope"}},
                {"name": "a", "when": {"code": "high_null_rate"}},
            ]
        }
    )
    outcome, _ = evaluate_policy(run_of(nulls), policy)
    assert [(o.rule.name, o.triggered) for o in outcome.rules] == [
        ("b", False),
        ("a", True),
    ]


@pytest.mark.parametrize(
    ("then", "exit_code"),
    [
        ("annotate", 0),
        ("warn", 0),
        ({"notify": "slack:#data-quality"}, 0),
        ("fail", 1),
        (["warn", "fail"], 1),
    ],
)
def test_only_fail_sets_the_exit_code(nulls, then, exit_code):
    policy = load_policy(
        {"rules": [{"name": "r", "when": {"code": "high_null_rate"}, "then": then}]}
    )
    outcome, _ = evaluate_policy(run_of(nulls), policy)
    assert outcome.exit_code == exit_code


def test_a_fail_rule_that_matches_nothing_does_not_fail(nulls):
    policy = load_policy(
        {"rules": [{"name": "r", "when": {"code": "nope"}, "then": "fail"}]}
    )
    assert evaluate_policy(run_of(nulls), policy)[0].exit_code == 0


def test_no_rules_exit_zero(nulls):
    outcome, notes = evaluate_policy(run_of(nulls), Policy())
    assert (outcome.rules, outcome.exit_code, notes) == ((), 0, ())


def test_the_outcome_is_accepted_by_the_run(nulls):
    run = run_of(nulls)
    policy = load_policy(
        {"rules": [{"name": "r", "when": {"code": "high_null_rate"}, "then": "fail"}]}
    )
    outcome, _ = evaluate_policy(run, policy)
    checked = type(run).model_validate(run.model_dump() | {"policy": outcome})
    assert checked.policy == outcome
