# dsbf/service/views.py
"""Responses that are views of the contracts rather than contracts themselves."""

import math
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

import pyarrow as pa
from pydantic import Field

from dsbf.model.base import Contract
from dsbf.model.evidence import SamplingInfo
from dsbf.store import FindingCount, RunHeader, StoredArtifact


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
