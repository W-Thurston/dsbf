# dsbf/store/jobs.py
"""Profiling jobs: what was asked, how far it got and how it ended."""

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field

from dsbf.config import ProfileConfig
from dsbf.model import CheckStatus
from dsbf.model.base import Contract


class JobState(StrEnum):
    """Where a job is. The last four are final."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    @property
    def final(self) -> bool:
        return self not in (JobState.QUEUED, JobState.RUNNING)


class JobRequest(Contract):
    """What to profile and with which settings."""

    source: str = Field(
        min_length=1,
        description="The path of a local CSV, TSV, Parquet, Excel, JSON or NDJSON "
        "file.",
    )
    name: str | None = Field(
        default=None, description="The name to give the dataset; unset is the file's."
    )
    settings: ProfileConfig = Field(
        default_factory=ProfileConfig,
        description="The settings `dsbf check` would build, `dataset` being the key "
        "that groups the run with earlier ones.",
    )


class Job(Contract):
    """One profiling job."""

    job_id: UUID
    state: JobState
    request: JobRequest
    requested_at: AwareDatetime
    started_at: AwareDatetime | None = Field(
        description="When the worker took the job; null while it is queued."
    )
    finished_at: AwareDatetime | None = Field(
        description="When the job reached a final state."
    )
    run_id: UUID | None = Field(description="The stored run, once the job succeeded.")
    error: str | None = Field(
        description="Why the job failed, as `dsbf check` would say it; also set when "
        "a job was cancelled or interrupted, with what happened."
    )
    last_seq: int = Field(
        ge=0, description="The number of the newest event, 0 if none."
    )


class JobEvent(Contract):
    """One step of a job's progress."""

    seq: int = Field(ge=1, description="The event's number within its job.")
    at: AwareDatetime
    kind: Literal["check_started", "note", "check_finished", "job_finished"]
    check_id: str | None = Field(description="The check; null on `job_finished`.")
    position: int | None = Field(description="The check's place in the plan, from 1.")
    total: int | None = Field(description="Checks in the plan.")
    status: CheckStatus | None = Field(description="Set on `check_finished`.")
    code: str | None = Field(description="A note's code.")
    message: str | None = Field(
        description="A note's message, or on `job_finished` the job's error."
    )
    state: JobState | None = Field(
        description="Set on `job_finished`: the final state."
    )
    run_id: UUID | None = Field(
        description="Set on `job_finished` when the run was stored."
    )
