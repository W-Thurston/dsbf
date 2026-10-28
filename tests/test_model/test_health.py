# tests/test_model/test_health.py
"""The run summary counts every status, explains every problem and renders as text."""

import pytest

from dsbf.model import CheckStatus, RunHealth, Severity, StatusCounts

from .builders import make_finding, make_note, make_result, round_trip

pytestmark = pytest.mark.unit

HEADER = "DSBF run 7f3c…  titanic  (891 × 15, full scan)   61 checks"
POINTER = "see findings.parquet / dashboard"
BUDGET = "Comparison budget exceeded after 300s → partial results not stored."
HINT = "lower max_rows or install dsbf[text]."
ERROR_TEXT = f"{BUDGET} Hint: {HINT}"
SKEWNESS = "Requires 'skewness' (error)."

# The summary shown in REFACTOR_PLAN §7.1.
SECTION_7_1 = "\n".join(
    [
        HEADER,
        "  ok 54 · not applicable 3 · unavailable 1 · skipped 2 · error 1",
        "  ✖ error        fuzzy_duplicates      " + ERROR_TEXT,
        "  ↷ skipped      ml_readiness          Requires 'skewness' (error).",
        "  ↷ skipped      outlier_report        Requires 'skewness' (error).",
        "  ⦸ unavailable  stationarity          Install dsbf[timeseries].",
        f"  ⚠ 14 findings at warn or above → {POINTER}",
    ]
)


def warn_findings(count, check_id="summarize_nulls"):
    return tuple(
        make_finding(check_id=check_id, code=f"code_{i}", severity=Severity.WARN)
        for i in range(count)
    )


def section_7_1_results():
    results = {}
    for i in range(54):
        findings = warn_findings(14, f"ok_{i}") if i == 0 else ()
        results[f"ok_{i}"] = make_result(check_id=f"ok_{i}", findings=findings)
    for i in range(3):
        results[f"na_{i}"] = make_result(CheckStatus.NOT_APPLICABLE, check_id=f"na_{i}")
    results["stationarity"] = make_result(
        CheckStatus.UNAVAILABLE,
        check_id="stationarity",
        notes=(make_note(code="missing_extra", message="Install dsbf[timeseries]."),),
    )
    for name in ("ml_readiness", "outlier_report"):
        results[name] = make_result(
            CheckStatus.SKIPPED,
            check_id=name,
            notes=(
                make_note(
                    message="Requires 'skewness' (error).", blocked_by=("skewness",)
                ),
            ),
        )
    results["fuzzy_duplicates"] = make_result(
        CheckStatus.ERROR,
        check_id="fuzzy_duplicates",
        notes=(
            make_note(
                code="check_error",
                message=BUDGET,
                hint=HINT,
            ),
        ),
    )
    return results


def test_counts_and_issues_come_from_the_results():
    health = RunHealth.from_results(section_7_1_results())

    assert health.counts == StatusCounts(
        ok=54, not_applicable=3, unavailable=1, skipped=2, error=1
    )
    assert health.counts.total == 61
    assert health.findings_at_warn_or_above == 14
    assert [(i.status, i.check_id) for i in health.issues] == [
        (CheckStatus.ERROR, "fuzzy_duplicates"),
        (CheckStatus.SKIPPED, "ml_readiness"),
        (CheckStatus.SKIPPED, "outlier_report"),
        (CheckStatus.UNAVAILABLE, "stationarity"),
        (CheckStatus.NOT_APPLICABLE, "na_0"),
        (CheckStatus.NOT_APPLICABLE, "na_1"),
        (CheckStatus.NOT_APPLICABLE, "na_2"),
    ]
    assert health.issues[1].blocked_by == ("skewness",)
    assert round_trip(health) == health


def test_not_applicable_is_listed_with_its_reason_but_not_printed():
    note = make_note(code="no_numeric_columns", message="No numeric columns.")
    health = RunHealth.from_results(
        {"c": make_result(CheckStatus.NOT_APPLICABLE, check_id="c", notes=(note,))}
    )

    assert health.counts.not_applicable == 1
    assert [(i.check_id, i.message) for i in health.issues] == [
        ("c", "No numeric columns.")
    ]
    assert "No numeric columns." not in health.render()


def test_renders_the_summary_in_the_design_document():
    health = RunHealth.from_results(section_7_1_results())

    assert health.render(header=HEADER, pointer=POINTER) == SECTION_7_1


def test_a_timeout_is_an_issue_after_errors():
    results = {
        "a": make_result(CheckStatus.TIMEOUT, check_id="a"),
        "b": make_result(CheckStatus.ERROR, check_id="b"),
    }

    health = RunHealth.from_results(results)

    assert [i.check_id for i in health.issues] == ["b", "a"]
    assert "timeout 1" in health.render()


def test_only_findings_at_warn_or_above_are_counted():
    findings = (
        make_finding(code="a", severity=Severity.INFO),
        make_finding(code="b", severity=Severity.NOTICE),
        make_finding(code="c", severity=Severity.WARN),
        make_finding(code="d", severity=Severity.ERROR),
    )

    health = RunHealth.from_results({"c": make_result(findings=findings)})

    assert health.findings_at_warn_or_above == 2


def test_renders_one_finding_in_the_singular_and_none_plainly():
    one = RunHealth.from_results({"c": make_result(findings=warn_findings(1))})
    none = RunHealth.from_results({"c": make_result(findings=())})

    assert "⚠ 1 finding at warn or above" in one.render()
    assert "→" not in one.render()
    assert none.render() == "  ok 1\n  no findings at warn or above"


def test_a_long_check_id_widens_the_column():
    long_id = "detect_bimodal_distribution_with_a_long_name"
    health = RunHealth.from_results(
        {
            "a": make_result(CheckStatus.SKIPPED, check_id=long_id),
            "b": make_result(CheckStatus.SKIPPED, check_id="short"),
        }
    )

    lines = health.render().splitlines()[1:3]
    starts = {line.index("Requires") for line in lines}
    assert len(starts) == 1


def test_an_empty_run_renders():
    assert RunHealth.from_results({}).render() == "  \n  no findings at warn or above"
