# dsbf/store/repository.py
"""What a run repository does, and the views it returns."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, Protocol
from uuid import UUID

import pyarrow as pa
from pydantic import AwareDatetime, Field

from dsbf.model import (
    CheckResult,
    CheckStatus,
    ColumnCounts,
    ColumnProfile,
    Confidence,
    DatasetManifest,
    Dimension,
    Finding,
    Note,
    PlanManifest,
    PolicyOutcome,
    Run,
    RunHealth,
    SemanticType,
    Severity,
)
from dsbf.model.base import Contract
from dsbf.model.evidence import SamplingInfo
from dsbf.store.jobs import Job, JobEvent, JobRequest, JobState

FindingSort = Literal["severity", "check", "code"]
"""How ``find_findings`` orders a run's findings."""


class StoreError(Exception):
    """The store cannot do what was asked."""


class JobNotFoundError(StoreError):
    """No job has this id."""


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


class DatasetSummary(Contract):
    """A dataset's history in brief: how many runs, and the latest."""

    key: str
    name: str = Field(description="The name the latest run gave the dataset.")
    n_runs: int = Field(ge=1)
    last_run_id: UUID
    last_run_at: AwareDatetime
    last_max_severity: Severity | None = Field(
        description="The highest severity in the latest run; null when it has none."
    )


class RunHeader(Contract):
    """A stored run without its results and profile: what identifies it and how it
    went."""

    schema_version: str
    run_id: UUID
    started_at: AwareDatetime
    finished_at: AwareDatetime
    dataset: DatasetManifest
    plan: PlanManifest
    health: RunHealth
    notes: tuple[Note, ...] = ()
    policy: PolicyOutcome | None = None


class FindingCount(Contract):
    """How many findings of one dimension and severity a run has."""

    dimension: Dimension
    severity: Severity
    count: int = Field(ge=1)


class ColumnSummary(Contract):
    """A column in brief: enough to list and filter a run's columns."""

    name: str
    position: int = Field(ge=0, description="Place in the dataset's column order.")
    semantic_type: SemanticType
    physical_dtype: str
    counts: ColumnCounts


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

    def save(self, run: Run, *, snapshot_rows: int | None = None) -> None:
        """Store ``run`` with its artifacts and a snapshot of its data.

        ``snapshot_rows`` caps the snapshot for this run (0 keeps none); unset, the
        repository's own cap applies.

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

    def list_datasets(
        self, *, limit: int = 50, offset: int = 0
    ) -> Page[DatasetSummary]:
        """Datasets with at least one run, the most recently run first."""

    def get_header(self, run_id: UUID) -> RunHeader:
        """A run's identity, health and policy, read without its results.

        Raises:
            RunNotFoundError: no such run.
        """

    def finding_counts(self, run_id: UUID) -> tuple[FindingCount, ...]:
        """A run's finding counts by dimension and severity, in that order.

        Raises:
            RunNotFoundError: no such run.
        """

    def get_column_profile(self, run_id: UUID, column: str) -> ColumnProfile:
        """One column's profile, read without validating the others.

        Raises:
            RunNotFoundError: no such run, or the run has no such column.
        """

    def list_columns(self, run_id: UUID) -> tuple[ColumnSummary, ...]:
        """Every profiled column in the dataset's order, without its statistics.

        Raises:
            RunNotFoundError: no such run.
        """

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
        min_confidence: Confidence | None = None,
        code: str | None = None,
        check: str | None = None,
        column: str | None = None,
        sort: FindingSort = "severity",
        limit: int = 100,
        offset: int = 0,
    ) -> Page[Finding]:
        """A run's findings that meet every filter (a pair matches either column).

        ``sort`` is ``severity`` (highest first), ``check`` or ``code``; ties break
        by check id, then id."""

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

    def create_job(self, request: JobRequest) -> Job:
        """Queue a job."""

    def get_job(self, job_id: UUID) -> Job:
        """The job.

        Raises:
            JobNotFoundError: no such job.
        """

    def list_jobs(
        self, *, state: JobState | None = None, limit: int = 50, offset: int = 0
    ) -> Page[Job]:
        """Jobs, newest request first, optionally only those in ``state``."""

    def start_job(self, job_id: UUID) -> Job:
        """Mark a queued job running."""

    def append_job_event(
        self,
        job_id: UUID,
        kind: str,
        *,
        check_id: str | None = None,
        position: int | None = None,
        total: int | None = None,
        status: CheckStatus | None = None,
        code: str | None = None,
        message: str | None = None,
    ) -> JobEvent:
        """Record a progress event, numbered after the job's last."""

    def finish_job(
        self,
        job_id: UUID,
        state: JobState,
        *,
        run_id: UUID | None = None,
        error: str | None = None,
    ) -> Job:
        """End a job in a final ``state``, with its closing event."""

    def job_events(
        self, job_id: UUID, *, after: int = 0, limit: int = 500
    ) -> tuple[JobEvent, ...]:
        """The job's events numbered after ``after``, in order.

        Raises:
            JobNotFoundError: no such job.
        """

    def interrupt_unfinished(self) -> int:
        """End every queued or running job as ``interrupted``; how many there were."""

    def get_settings_overlay(self, dataset_key: str) -> dict[str, Any] | None:
        """The settings saved for a dataset, as the fields the user set; ``None``
        when nothing is saved."""

    def set_settings_overlay(self, dataset_key: str, overlay: dict[str, Any]) -> None:
        """Save a dataset's settings, replacing any earlier ones."""

    def delete_settings_overlay(self, dataset_key: str) -> bool:
        """Remove a dataset's saved settings; whether there were any."""


__all__ = [
    "DatasetSummary",
    "FindingCount",
    "FindingOccurrence",
    "FindingSort",
    "GcResult",
    "JobNotFoundError",
    "Page",
    "RunExistsError",
    "RunHeader",
    "RunNotFoundError",
    "RunRepository",
    "SNAPSHOT_REF",
    "RunSummary",
    "Snapshot",
    "StoreError",
    "StoredArtifact",
]
