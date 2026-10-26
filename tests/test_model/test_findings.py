"""The findings view: filtering with the policy vocabulary, and the dataframe forms."""

import pytest

from dsbf.model import Dimension, Findings, Severity, Subject
from tests.test_model.builders import (
    make_evidence,
    make_finding,
    make_result,
    make_run,
)

pytestmark = pytest.mark.unit


def finding(code, column, severity=Severity.WARN, subject=None, **overrides):
    return make_finding(
        code=code,
        check_id=overrides.pop("check_id", "null_rate"),
        subject=subject or Subject.column(column),
        severity=severity,
        **overrides,
    )


@pytest.fixture
def run():
    findings = (
        finding(
            "high_null_rate",
            "age",
            Severity.ERROR,
            evidence=make_evidence(metrics={"null_share": 0.4}),
        ),
        finding("low_null_rate", "fare", Severity.INFO, tags=("minor",)),
        finding(
            "correlated",
            "age",
            subject=Subject.column_pair("age", "fare"),
            check_id="pairwise",
            dimension=Dimension.REDUNDANCY,
        ),
        finding("empty", "x", subject=Subject.dataset(), check_id="dataset_check"),
    )
    by_check: dict[str, list] = {}
    for item in findings:
        by_check.setdefault(item.check_id, []).append(item)
    return make_run(
        [make_result(check_id=c, findings=tuple(f)) for c, f in by_check.items()]
    )


def codes(view):
    return [f.code for f in view]


def test_a_run_gives_an_ordered_view_that_still_behaves_as_a_tuple(run):
    assert isinstance(run.findings, Findings)
    assert isinstance(run.findings, tuple)
    assert len(run.findings) == 4
    assert run.findings[0].severity is Severity.ERROR


def test_no_conditions_keep_every_finding(run):
    assert run.findings.filter() == run.findings


def test_each_key_narrows_the_view(run):
    view = run.findings
    assert codes(view.filter(code="correlated")) == ["correlated"]
    assert codes(view.filter(check="null_rate")) == ["high_null_rate", "low_null_rate"]
    assert codes(view.filter(dimension="redundancy")) == ["correlated"]
    assert codes(view.filter(tag="minor")) == ["low_null_rate"]
    assert codes(view.filter(severity=">=error")) == ["high_null_rate"]
    assert codes(view.filter(metric={"null_share": ">0.3"})) == ["high_null_rate"]


def test_a_column_matches_a_pair_on_either_side_and_never_the_dataset(run):
    assert sorted(codes(run.findings.filter(column="fare"))) == [
        "correlated",
        "low_null_rate",
    ]
    assert codes(run.findings.filter(column="x")) == []


def test_filters_chain_and_keep_the_order(run):
    view = run.findings.filter(column="age").filter(severity=">=warn")
    assert isinstance(view, Findings)
    assert codes(view) == ["high_null_rate", "correlated"]


def test_a_value_a_rule_would_reject_is_refused(run):
    with pytest.raises(ValueError):
        run.findings.filter(severity=">=loud")
    with pytest.raises(ValueError):
        run.findings.filter(colour="red")


def test_to_polars_has_a_row_per_finding_and_flat_columns(run):
    frame = run.findings.to_polars()
    assert frame.height == 4
    assert frame["code"].to_list()[0] == "high_null_rate"
    pair = frame.filter(frame["code"] == "correlated")
    assert pair["columns"].to_list() == [["age", "fare"]]
    assert pair["subject_kind"].to_list() == ["column_pair"]


def test_an_empty_view_keeps_the_columns(run):
    full = run.findings.to_polars()
    empty = run.findings.filter(code="nothing").to_polars()
    assert empty.height == 0
    assert empty.schema == full.schema


def test_to_pandas_matches_the_polars_table(run):
    frame = run.findings.to_pandas()
    assert list(frame["code"]) == run.findings.to_polars()["code"].to_list()


def test_to_pandas_without_pandas_names_the_extra(run, monkeypatch):
    monkeypatch.setattr("dsbf.model.findings.find_spec", lambda name: None)
    with pytest.raises(ImportError, match=r"dsbf\[pandas\]"):
        run.findings.to_pandas()


def test_the_notebook_display_shows_health_and_top_findings(run):
    html = run._repr_html_()
    assert "Top findings" in html
    assert "high_null_rate" not in html  # titles, not codes
    assert run.findings[0].title in html


def test_the_notebook_display_escapes_names_and_titles():
    hostile = make_finding(
        code="x",
        check_id="null_rate",
        subject=Subject.column("<script>alert(1)</script>|[b]"),
        title="<img src=x onerror=1>",
    )
    run = make_run([make_result(check_id="null_rate", findings=(hostile,))])
    html = run._repr_html_()
    assert "<script>" not in html
    assert "<img" not in html
    assert "&lt;script&gt;" in html


def test_the_notebook_display_says_when_there_are_no_findings():
    assert (
        "No findings"
        in make_run([make_result(check_id="null_rate", findings=())])._repr_html_()
    )


def test_the_notebook_display_caps_the_table():
    many = tuple(
        make_finding(
            code=f"c{i}", check_id="null_rate", subject=Subject.column(f"c{i}")
        )
        for i in range(15)
    )
    html = make_run([make_result(check_id="null_rate", findings=many)])._repr_html_()
    assert "5 more findings not shown" in html
