# dsbf/model/diff.py
"""What changed between two runs."""

from pydantic import Field

from dsbf.model.base import Contract, FiniteFloat, Scalar
from dsbf.model.finding import Finding
from dsbf.model.vocabulary import CheckStatus, Severity

DIFF_SCHEMA_VERSION = "1.0.0"


class MetricChange(Contract):
    """A metric of a finding that moved beyond the tolerance."""

    name: str
    base: Scalar
    head: Scalar


class FindingChange(Contract):
    """A finding present in both runs whose severity or metrics changed."""

    id: str
    code: str
    check_id: str
    title: str
    base_severity: Severity
    head_severity: Severity
    metrics: tuple[MetricChange, ...] = ()


class StatusChange(Contract):
    """A check whose status differs between the runs; null when it ran in one only."""

    check_id: str
    base: CheckStatus | None
    head: CheckStatus | None


class RunDiff(Contract):
    """Findings and check statuses of a head run against a base run."""

    schema_version: str = DIFF_SCHEMA_VERSION
    base_run_id: str
    head_run_id: str
    same_dataset: bool = Field(
        description="False when the runs have different dataset keys, or either has "
        "none; the other fields still describe the differences."
    )
    tolerance: FiniteFloat = Field(
        description="Relative change a numeric metric must exceed to count as changed."
    )
    new: tuple[Finding, ...] = Field(description="In the head run only.")
    resolved: tuple[Finding, ...] = Field(description="In the base run only.")
    changed: tuple[FindingChange, ...]
    status_changes: tuple[StatusChange, ...]

    def render(self) -> str:
        """A plain-text summary: the counts, then each difference on a line."""
        lines = [f"base {self.base_run_id}", f"head {self.head_run_id}"]
        if not self.same_dataset:
            lines.append("note: the runs are not of the same dataset")
        lines.append(
            f"{len(self.new)} new, {len(self.resolved)} resolved, "
            f"{len(self.changed)} changed findings; "
            f"{len(self.status_changes)} check status changes"
        )
        lines += [f"+ {f.severity.value:<6} {f.code}  {f.title}" for f in self.new]
        lines += [f"- {f.severity.value:<6} {f.code}  {f.title}" for f in self.resolved]
        for change in self.changed:
            moves = ", ".join(
                f"{m.name} {_short(m.base)} -> {_short(m.head)}" for m in change.metrics
            )
            severity = (
                f"{change.base_severity.value} -> {change.head_severity.value}"
                if change.base_severity != change.head_severity
                else change.head_severity.value
            )
            lines.append(
                f"~ {severity}  {change.code}  {change.title}  {moves}".rstrip()
            )
        lines += [
            f"! {c.check_id}: {c.base.value if c.base else 'absent'} -> "
            f"{c.head.value if c.head else 'absent'}"
            for c in self.status_changes
        ]
        return "\n".join(lines)


def _short(value: Scalar) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)
