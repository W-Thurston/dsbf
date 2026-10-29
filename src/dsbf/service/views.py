# dsbf/service/views.py
"""Responses that are views of the contracts rather than contracts themselves."""

import math
from collections.abc import Sequence
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from uuid import UUID

import pyarrow as pa
from pydantic import AwareDatetime, Field

from dsbf.model.base import Contract
from dsbf.model.evidence import SamplingInfo
from dsbf.store import FindingCount, Job, JobEvent, RunHeader, StoredArtifact


class RunOverview(Contract):
    """A run without its results: what it is, how it went and what it found."""

    header: RunHeader
    finding_counts: tuple[FindingCount, ...] = Field(
        description="Findings by dimension, then severity (highest first)."
    )
    n_findings: int = Field(ge=0)
    artifacts: tuple[StoredArtifact, ...] = Field(
        description="The files stored for the run, the data snapshot included."
    )


class LineageEntry(Contract):
    """Where one run of a dataset read its data from, and whether that moved."""

    run_id: UUID
    started_at: AwareDatetime
    location: str | None = Field(description="The source's path, URL or table name.")
    fingerprint: str
    n_rows: int = Field(ge=0, description="Rows in the source, before any sampling.")
    location_changed: bool = Field(
        description="The location differs from the previous run's; false for the "
        "first run."
    )
    fingerprint_changed: bool = Field(
        description="The content differs from the previous run's; false for the "
        "first run."
    )


def lineage_entries(
    headers: Sequence[RunHeader], *, previous: RunHeader | None
) -> tuple[LineageEntry, ...]:
    """Lineage entries for ``headers`` (newest first), each flagged against the run
    before it; ``previous`` is the run before the oldest header, if any."""
    older = [*headers[1:], previous]
    return tuple(
        LineageEntry(
            run_id=header.run_id,
            started_at=header.started_at,
            location=header.dataset.source.location,
            fingerprint=header.dataset.fingerprint,
            n_rows=header.dataset.n_rows,
            location_changed=before is not None
            and before.dataset.source.location != header.dataset.source.location,
            fingerprint_changed=before is not None
            and before.dataset.fingerprint != header.dataset.fingerprint,
        )
        for header, before in zip(headers, older, strict=True)
    )


class ArtifactRows(Contract):
    """A range of rows of a table artifact (or of the data snapshot)."""

    ref: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...] = Field(
        description="Row-major values; NaN and infinity are null, as JSON has neither."
    )
    total: int = Field(ge=0, description="Rows in the whole table.")
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    sampling: SamplingInfo | None = Field(
        description="For the snapshot, how its rows were drawn; null otherwise."
    )


class ArtifactValue(Contract):
    """A JSON artifact."""

    ref: str
    value: Any


def json_safe(value: object) -> object:
    """``value`` as something JSON holds: dates as text, non-finite numbers null."""
    match value:
        case float() if not math.isfinite(value):
            return None
        case datetime() | date() | time():
            return value.isoformat()
        case Decimal():
            return str(value)
        case bytes():
            return value.hex()
        case list() | tuple():
            return [json_safe(item) for item in value]
        case dict():
            return {str(key): json_safe(item) for key, item in value.items()}
        case _:
            return value


def table_rows(table: pa.Table) -> tuple[tuple[Any, ...], ...]:
    columns = [[json_safe(v) for v in column.to_pylist()] for column in table.columns]
    return tuple(zip(*columns, strict=True))


class JobProgress(Contract):
    """A job and the events after the cursor the caller gave."""

    job: Job
    events: tuple[JobEvent, ...] = Field(description="Events numbered after `after`.")
    next_after: int = Field(
        ge=0, description="Pass this as `after` to get the events that follow."
    )
