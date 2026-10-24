"""Read a `Run` for the archetype tests: findings by check, levels by dimension."""

from __future__ import annotations

from typing import Any

from dsbf.model import CheckStatus, Finding, Run, Severity

FAILED = (CheckStatus.ERROR, CheckStatus.TIMEOUT)


def findings(
    run: Run,
    check_id: str,
    code: str | None = None,
    at_least: Severity = Severity.INFO,
) -> list[Finding]:
    """The findings of ``check_id`` with ``code``, at ``at_least`` or more."""
    return [
        f
        for f in run.results[check_id].findings
        if (code is None or f.code == code) and f.severity >= at_least
    ]


def flagged(
    run: Run,
    check_id: str,
    code: str | None = None,
    at_least: Severity = Severity.INFO,
) -> set[str]:
    """The columns named by the findings of ``check_id`` (all columns of each)."""
    return {
        column
        for f in findings(run, check_id, code, at_least)
        for column in f.subject.columns
    }


def output(run: Run, check_id: str) -> dict[str, Any]:
    result = run.results[check_id]
    assert result.output is not None, f"{check_id} ended {result.status.value}"
    return result.output


def quality(run: Run, dimension: str) -> dict[str, Any]:
    """One quality dimension of the `quality_dimensions` output."""
    return output(run, "quality_dimensions")["dimensions"][dimension]


def readiness(run: Run) -> dict[str, Any]:
    return output(run, "ml_readiness")


def statuses(run: Run, status: CheckStatus) -> set[str]:
    return {k for k, r in run.results.items() if r.status == status}


def column_types(run: Run) -> dict[str, str]:
    return {c.name: c.semantic_type for c in run.profile.columns}
