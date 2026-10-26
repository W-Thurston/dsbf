"""``dsbf.profile(policy=...)``: the rules are evaluated over a real run."""

import warnings

import pytest

import dsbf
from dsbf.model import Run
from dsbf.policy import PolicyError
from tests.test_library.frames import build_frame
from tests.test_model.test_policy import EXAMPLE

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def quiet_libraries():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def frame():
    return build_frame()


@pytest.fixture(scope="module")
def plain(frame) -> Run:
    return dsbf.profile(frame, depth="quick")


def test_without_a_policy_the_run_has_no_outcome(plain):
    assert plain.policy is None


def test_the_documented_example_produces_the_documented_outcome(frame, tmp_path, plain):
    path = tmp_path / "policy.yaml"
    path.write_text(EXAMPLE, encoding="utf-8")
    run = dsbf.profile(frame, depth="quick", policy=path)
    assert run.policy is not None
    outcomes = {o.rule.name: o for o in run.policy.rules}
    assert list(outcomes) == [
        "block-on-leakage",
        "sparse-columns",
        "check-errors-are-notes",
        "tell-the-team",
    ]
    # Nothing here leaks, no null share passes 0.3, and no check errored.
    assert not any(o.triggered for o in outcomes.values())
    assert run.policy.exit_code == 0
    assert [n.code for n in run.notes] == [
        *(n.code for n in plain.notes),
        "policy_no_previous_run",
    ]


def test_a_fail_rule_sets_the_exit_code_and_the_run_still_completes(frame, plain):
    policy = {
        "rules": [
            {"name": "heavy-tails", "when": {"code": "heavy_tails"}, "then": "fail"},
            {"name": "some-nulls", "when": {"metric": {"null_share": ">0.1"}}},
        ]
    }
    run = dsbf.profile(frame, depth="quick", policy=policy)
    assert run.policy is not None
    assert run.policy.exit_code == 1
    heavy, nulls = run.policy.rules
    ids = {f.id: f for f in run.findings}
    assert [ids[i].subject.columns for i in heavy.finding_ids] == [("fare",)]
    assert {ids[i].subject.columns[0] for i in nulls.finding_ids} == {"age"}
    assert set(run.results) == set(plain.results)


def test_the_result_validates_as_a_run(frame):
    policy = {"rules": [{"name": "r", "when": {"dimension": "completeness"}}]}
    run = dsbf.profile(frame, depth="quick", policy=policy)
    assert Run.model_validate_json(run.model_dump_json()) == run


def test_new_since_uses_the_previous_run(frame, plain):
    policy = {"rules": [{"name": "r", "when": {"new_since": "previous_run"}}]}
    same = dsbf.profile(frame, depth="quick", policy=policy, previous=plain)
    assert same.policy is not None
    assert not same.policy.rules[0].triggered
    assert "policy_no_previous_run" not in {n.code for n in same.notes}
    changed = dsbf.profile(
        frame.drop("fare"), depth="quick", policy=policy, previous=plain
    )
    assert changed.policy is not None
    assert not changed.policy.rules[0].triggered
    other = dsbf.profile(frame.with_columns(extra=frame["age"]), depth="quick")
    grown = dsbf.profile(
        frame.with_columns(extra=frame["age"]),
        depth="quick",
        policy=policy,
        previous=plain,
    )
    assert grown.policy is not None
    new_ids = {f.id for f in other.findings} - {f.id for f in plain.findings}
    assert set(grown.policy.rules[0].finding_ids) == new_ids
    assert new_ids


def test_a_previous_run_without_a_policy_is_ignored(frame, plain):
    assert dsbf.profile(frame, depth="quick", previous=plain).policy is None


def test_a_bad_policy_fails_before_the_data_is_read():
    with pytest.raises(PolicyError, match=r"rules\[r\]"):
        dsbf.profile(
            "/nonexistent/data.csv",
            policy={"rules": [{"name": "r", "when": {"nope": 1}}]},
        )


def test_previous_can_be_the_path_of_a_stored_run(frame, tmp_path):
    first = dsbf.profile(frame, depth="quick")
    stored = first.write(tmp_path)["json"]
    policy = {"rules": [{"name": "new", "when": {"new_since": "previous_run"}}]}
    from_path = dsbf.profile(frame, depth="quick", policy=policy, previous=stored)
    from_run = dsbf.profile(frame, depth="quick", policy=policy, previous=first)
    assert from_path.policy is not None
    assert from_run.policy is not None
    assert from_path.policy.rules == from_run.policy.rules
    assert not any("previous run" in n.message for n in from_path.notes)


def test_an_unreadable_previous_path_fails_before_the_data_is_read(tmp_path):
    from dsbf.sinks import RunReadError

    policy = {"rules": [{"name": "new", "when": {"new_since": "previous_run"}}]}
    with pytest.raises(RunReadError, match="cannot read run"):
        dsbf.profile(
            tmp_path / "missing.csv", policy=policy, previous=tmp_path / "nope.json"
        )
