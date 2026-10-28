# dsbf/model/health.py
"""The run summary a reader sees without opening any log."""

from collections.abc import Mapping
from typing import Self

from pydantic import Field

from dsbf.model.base import Contract
from dsbf.model.check import CheckResult, Note
from dsbf.model.vocabulary import CheckStatus, Severity

# The statuses that carry a reason, in the order a reader should see them. "Not
# applicable" is expected, so it is listed for the dashboard but not printed.
_ISSUE_ORDER = (
    CheckStatus.ERROR,
    CheckStatus.TIMEOUT,
    CheckStatus.SKIPPED,
    CheckStatus.UNAVAILABLE,
    CheckStatus.NOT_APPLICABLE,
)
_MARKS = {
    CheckStatus.ERROR: "✖",
    CheckStatus.TIMEOUT: "✖",
    CheckStatus.SKIPPED: "↷",
    CheckStatus.UNAVAILABLE: "⦸",
}
_PRINTED = tuple(_MARKS)
_MIN_ID_WIDTH = 22
_STATUS_WIDTH = len("unavailable") + 2  # the longest status, two spaces before the id


class StatusCounts(Contract):
    """How many checks ended in each status."""

    ok: int = Field(default=0, ge=0)
    not_applicable: int = Field(default=0, ge=0)
    unavailable: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    error: int = Field(default=0, ge=0)
    timeout: int = Field(default=0, ge=0)

    def __getitem__(self, status: CheckStatus) -> int:
        return getattr(self, status.value)

    @property
    def total(self) -> int:
        return sum(self[status] for status in CheckStatus)


class HealthIssue(Contract):
    """One check that did not complete normally, and why."""

    check_id: str
    status: CheckStatus
    message: str
    hint: str | None = None
    blocked_by: tuple[str, ...] = ()


class RunHealth(Contract):
    """Counts and reasons by status, so a failure is never found by reading logs."""

    counts: StatusCounts
    issues: tuple[HealthIssue, ...] = Field(
        default=(),
        description=(
            "Checks that errored, timed out, were skipped, were unavailable or were"
            " not applicable, each with the reason."
        ),
    )
    findings_at_warn_or_above: int = Field(ge=0)

    @classmethod
    def from_results(cls, results: Mapping[str, CheckResult]) -> Self:
        counts = dict.fromkeys((status.value for status in CheckStatus), 0)
        issues: list[HealthIssue] = []
        at_warn = 0
        for result in results.values():
            counts[result.status.value] += 1
            at_warn += sum(f.severity >= Severity.WARN for f in result.findings)
            if result.status in _ISSUE_ORDER:
                note = result.notes[0]
                issues.append(
                    HealthIssue(
                        check_id=result.check_id,
                        status=result.status,
                        message=note.message,
                        hint=note.hint,
                        blocked_by=note.blocked_by,
                    )
                )
        issues.sort(key=lambda i: (_ISSUE_ORDER.index(i.status), i.check_id))
        return cls(
            counts=StatusCounts(**counts),
            issues=tuple(issues),
            findings_at_warn_or_above=at_warn,
        )

    def render(
        self,
        header: str | None = None,
        pointer: str | None = None,
        notes: tuple[Note, ...] = (),
    ) -> str:
        """Plain text for a terminal or a log.

        ``header`` names the run (id, dataset, shape, check count), which the run
        knows and this object does not. ``pointer`` says where the findings are.
        ``notes`` are the run's own, which belong to no check.
        """
        lines = [header] if header else []
        lines.append(
            "  "
            + " · ".join(
                f"{status.value.replace('_', ' ')} {self.counts[status]}"
                for status in CheckStatus
                if self.counts[status]
            )
        )
        printed = [i for i in self.issues if i.status in _PRINTED]
        id_width = max([_MIN_ID_WIDTH, *(len(i.check_id) + 2 for i in printed)])
        for issue in printed:
            text = issue.message + (f" Hint: {issue.hint}" if issue.hint else "")
            lines.append(
                f"  {_MARKS[issue.status]} {issue.status.value:<{_STATUS_WIDTH}}"
                f"{issue.check_id:<{id_width}}{text}"
            )
        for note in notes:
            text = note.message + (f" Hint: {note.hint}" if note.hint else "")
            lines.append(f"  ! {note.code}: {text}")
        count = self.findings_at_warn_or_above
        if count:
            noun = "finding" if count == 1 else "findings"
            line = f"  ⚠ {count} {noun} at warn or above"
            lines.append(line + (f" → {pointer}" if pointer else ""))
        else:
            lines.append("  no findings at warn or above")
        return "\n".join(lines)
