# dsbf/store/repository.py
"""What a run repository does, and the views it returns."""

from typing import Protocol
from uuid import UUID

from pydantic import AwareDatetime, Field

from dsbf.model import CheckResult, Dimension, Finding, Run, Severity
from dsbf.model.base import Contract


class StoreError(Exception):
    """The store cannot do what was asked."""


class RunExistsError(StoreError):
    """A run with this id is already stored."""


class RunNotFoundError(StoreError):
    """No stored run, result or finding has this id."""


class Page[T](Contract):
    """One page of a list, and how long the whole list is."""

    items: tuple[T, ...]
    total: int = Field(ge=0, description="Items in the whole list, across pages.")
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class RunSummary(Contract):
    """A stored run without its results."""

    run_id: UUID
    dataset_key: str | None = Field(
        description="The dataset the run belongs to; null for an ungrouped run."
    )
    dataset_name: str
    started_at: AwareDatetime
    finished_at: AwareDatetime
    n_rows: int
    n_columns: int
    n_findings: int
    max_severity: Severity | None = Field(
        description="The highest severity among the findings; null when there are none."
    )


class FindingOccurrence(Contract):
    """A finding as one run of a dataset recorded it."""

    run_id: UUID
    started_at: AwareDatetime
    finding: Finding


class RunRepository(Protocol):
    """Where runs are kept. Implementations differ in storage, not behaviour."""

    def save(self, run: Run) -> None:
        """Store ``run`` with its artifacts.

        Raises:
            RunExistsError: a run with this id is stored.
        """

    def get(self, run_id: UUID) -> Run:
        """The stored run, equal to the one saved, its artifacts readable.

        Raises:
            RunNotFoundError: no such run.
        """

    def list_runs(
        self, *, dataset_key: str | None = None, limit: int = 50, offset: int = 0
    ) -> Page[RunSummary]:
        """Runs, newest first, of one dataset or all."""

    def get_result(self, run_id: UUID, check_id: str) -> CheckResult:
        """One check's result, read without the rest of the run."""

    def get_finding(self, run_id: UUID, finding_id: str) -> Finding:
        """One finding, read without the rest of the run."""

    def find_findings(
        self,
        run_id: UUID,
        *,
        dimension: Dimension | None = None,
        min_severity: Severity | None = None,
        code: str | None = None,
        column: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Page[Finding]:
        """A run's findings, highest severity first, then check id, then id."""

    def finding_history(
        self,
        dataset_key: str,
        *,
        finding_id: str | None = None,
        code: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Page[FindingOccurrence]:
        """Where a finding id (or every finding with a code) appeared in a dataset's
        runs, newest run first."""

    def delete(self, run_id: UUID) -> None:
        """Remove a run and its artifacts.

        Raises:
            RunNotFoundError: no such run.
        """


__all__ = [
    "FindingOccurrence",
    "Page",
    "RunExistsError",
    "RunNotFoundError",
    "RunRepository",
    "RunSummary",
    "StoreError",
]
