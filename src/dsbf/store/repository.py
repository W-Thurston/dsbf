# dsbf/store/repository.py
"""What a run repository does, and the views it returns."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol
from uuid import UUID

import pyarrow as pa
from pydantic import AwareDatetime, Field

from dsbf.model import CheckResult, Dimension, Finding, Run, Severity
from dsbf.model.base import Contract
from dsbf.model.evidence import SamplingInfo


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


SNAPSHOT_REF = "snapshot"
"""The ref of a run's data snapshot; a check's artifact ref always has a slash."""


class StoredArtifact(Contract):
    """One stored file of a run: an artifact, or the data snapshot."""

    ref: str
    media_type: str
    rows: int | None = Field(description="Rows of a table; null for a JSON value.")
    bytes: int = Field(ge=0)
    sha256: str
    sampling: SamplingInfo | None = Field(
        default=None,
        description="For the snapshot, how its rows were drawn from the analysed "
        "data; null when it holds all of them.",
    )


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The rows a run analysed, as stored, and how they were chosen."""

    table: pa.Table
    sampling: SamplingInfo | None


class FindingOccurrence(Contract):
    """A finding as one run of a dataset recorded it."""

    run_id: UUID
    started_at: AwareDatetime
    finding: Finding


class GcResult(Contract):
    """What a garbage collection removed (or would remove, for a dry run)."""

    runs: tuple[UUID, ...] = Field(description="Runs deleted, with their files.")
    orphans: tuple[str, ...] = Field(
        description="Run ids with artifact files but no stored run."
    )


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

    def artifacts(self, run_id: UUID) -> tuple[StoredArtifact, ...]:
        """The files stored for a run, snapshot included, in ref order.

        Raises:
            RunNotFoundError: no such run.
        """

    def read_artifact(self, run_id: UUID, ref: str) -> object:
        """The payload of artifact ``ref``: an Arrow table or a JSON value.

        Raises:
            RunNotFoundError: the run has no such artifact.
            ArtifactNotFoundError: the artifact is listed but its file is gone.
        """

    def read_artifact_rows(
        self, run_id: UUID, ref: str, start: int, stop: int
    ) -> pa.Table:
        """Rows ``start`` up to ``stop`` of a table artifact, without reading the
        rest of the file.

        Raises:
            RunNotFoundError: the run has no such table artifact.
            ArtifactNotFoundError: the artifact is listed but its file is gone.
        """

    def snapshot(self, run_id: UUID) -> Snapshot | None:
        """The stored data snapshot; null when none was kept.

        Raises:
            RunNotFoundError: no such run.
        """

    def delete(self, run_id: UUID) -> None:
        """Remove a run and its artifacts.

        Raises:
            RunNotFoundError: no such run.
        """

    def gc(
        self,
        *,
        keep: int | None = None,
        older_than: timedelta | None = None,
        dry_run: bool = False,
    ) -> GcResult:
        """Delete runs no rule keeps, and artifact files no run owns.

        ``keep`` retains the newest N runs of each dataset (ungrouped runs form one
        group); ``older_than`` retains runs that started more recently than that.
        A run any given rule retains stays. With neither, only orphaned files go.
        A ``dry_run`` changes nothing and reports what would go.
        """


__all__ = [
    "FindingOccurrence",
    "GcResult",
    "Page",
    "RunExistsError",
    "RunNotFoundError",
    "RunRepository",
    "SNAPSHOT_REF",
    "RunSummary",
    "Snapshot",
    "StoreError",
    "StoredArtifact",
]
