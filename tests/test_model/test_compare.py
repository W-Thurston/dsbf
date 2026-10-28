"""Comparing two runs: new, resolved and changed findings, and check statuses."""

import pytest

import dsbf
from dsbf.model import Evidence, Finding, Run, Severity, Subject
from tests.test_library.frames import build_frame

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def base() -> Run:
    return dsbf.profile(build_frame(), depth="quick", dataset="demo")


def with_findings(run: Run, findings: list[Finding]) -> Run:
    """``run`` whose findings are replaced: the first result holds all of them."""
    results = dict(run.results)
    for check_id in results:
        results[check_id] = results[check_id].model_copy(update={"findings": ()})
    first = next(iter(results))
    results[first] = results[first].model_copy(update={"findings": tuple(findings)})
    changed = run.model_copy(update={"results": results})
    changed.__dict__.pop("findings", None)  # the copy keeps the cached view
    return changed


def finding(code: str = "x", **update) -> Finding:
    subject = Subject(kind="dataset")
    return Finding.model_validate(
        {
            "id": Finding.make_id("c", subject, code),
            "check_id": "c",
            "check_version": "1",
            "code": code,
            "subject": subject,
            "dimension": "completeness",
            "severity": "warn",
            "confidence": "high",
            "title": "t",
            "message": "m",
            "evidence": {
                "metrics": {"rate": 0.2, "label": "a"},
                "n_used": 1,
                "n_total": 1,
                "method": "m",
            },
        }
        | update
    )


def test_a_run_compared_with_itself_has_no_differences(base):
    diff = dsbf.compare(base, base)
    assert not (diff.new or diff.resolved or diff.changed or diff.status_changes)
    assert diff.same_dataset


def test_findings_match_by_id_into_new_and_resolved(base):
    old = with_findings(base, [finding("a"), finding("b")])
    new = with_findings(base, [finding("b"), finding("c")])
    diff = dsbf.compare(old, new)
    assert [f.code for f in diff.new] == ["c"]
    assert [f.code for f in diff.resolved] == ["a"]
    assert diff.changed == ()


def test_a_severity_change_is_a_change(base):
    old = with_findings(base, [finding()])
    new = with_findings(base, [finding(severity="error")])
    (change,) = dsbf.compare(old, new).changed
    assert (change.base_severity, change.head_severity) == (
        Severity.WARN,
        Severity.ERROR,
    )


def evidence(rate: float) -> Evidence:
    return finding().evidence.model_copy(update={"metrics": {"rate": rate}})


@pytest.mark.parametrize(
    ("head_rate", "moved"), [(0.205, False), (0.3, True), (0.0, True)]
)
def test_a_metric_changes_beyond_the_tolerance(base, head_rate, moved):
    old = with_findings(base, [finding(evidence=evidence(0.2))])
    new = with_findings(base, [finding(evidence=evidence(head_rate))])
    diff = dsbf.compare(old, new)
    assert bool(diff.changed) is moved
    if moved:
        assert diff.changed[0].metrics[0].name == "rate"


def test_the_tolerance_is_adjustable(base):
    old = with_findings(base, [finding(evidence=evidence(0.2))])
    new = with_findings(base, [finding(evidence=evidence(0.205))])
    assert dsbf.compare(old, new, tolerance=0.01).changed


def test_a_status_change_names_the_check(base):
    check_id = next(iter(base.results))
    other = base.results[check_id].model_copy(update={"status": "skipped"})
    head = base.model_copy(update={"results": base.results | {check_id: other}})
    (change,) = dsbf.compare(base, head).status_changes
    assert change.check_id == check_id
    assert change.head is not None
    assert change.head.value == "skipped"


def test_runs_of_different_datasets_say_so(base):
    other = base.model_copy(
        update={"dataset": base.dataset.model_copy(update={"key": "elsewhere"})}
    )
    diff = dsbf.compare(base, other)
    assert not diff.same_dataset
    assert "not of the same dataset" in diff.render()


def test_a_negative_tolerance_is_refused(base):
    with pytest.raises(ValueError, match="tolerance"):
        dsbf.compare(base, base, tolerance=-1)


def test_a_p_value_moving_by_orders_of_magnitude_is_not_a_change(base):
    def with_p(p: float) -> Finding:
        metrics = {"p_value": p, "strongest_p_adjusted": p}
        return finding(
            evidence=finding().evidence.model_copy(update={"metrics": metrics})
        )

    old = with_findings(base, [with_p(1e-62)])
    new = with_findings(base, [with_p(1e-55)])
    assert dsbf.compare(old, new).changed == ()
