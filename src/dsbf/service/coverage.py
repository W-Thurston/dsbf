# dsbf/service/coverage.py
"""Which checks examined which columns, from the stored results."""

from pydantic import Field

from dsbf.model import CheckResult, CheckStatus
from dsbf.model.base import Contract


class CheckCoverage(Contract):
    """One check and the columns it analysed."""

    check_id: str
    status: CheckStatus
    columns: tuple[str, ...] = Field(
        description="The eligible columns the check analysed, after exclusions."
    )


class ColumnCoverage(Contract):
    """One column and how many completed checks examined it."""

    column: str
    examined_by: int = Field(ge=0)
    checks_completed: int = Field(
        ge=0, description="Checks that ended `ok`, whatever columns they read."
    )


class RunCoverage(Contract):
    """Coverage per check and per column. A check with no column to read, such as
    one that works on the whole table, examines none and is not counted against a
    column."""

    checks: tuple[CheckCoverage, ...]
    columns: tuple[ColumnCoverage, ...] = Field(
        description="Every profiled column, in the dataset's order."
    )


def build_coverage(
    results: dict[str, CheckResult], order: tuple[str, ...]
) -> RunCoverage:
    checks = tuple(
        CheckCoverage(check_id=r.check_id, status=r.status, columns=r.columns)
        for r in results.values()
    )
    completed = sum(1 for c in checks if c.status is CheckStatus.OK)
    examined = {name: 0 for name in order}
    for c in checks:
        if c.status is CheckStatus.OK:
            for name in set(c.columns):
                if name in examined:
                    examined[name] += 1
    return RunCoverage(
        checks=checks,
        columns=tuple(
            ColumnCoverage(
                column=n, examined_by=examined[n], checks_completed=completed
            )
            for n in order
        ),
    )
