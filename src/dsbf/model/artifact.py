# dsbf/model/artifact.py
"""Bulky data a check attaches to its result, addressed by a reference."""

import re
from typing import Any

from pydantic import Field, field_validator

from dsbf.model.base import Contract

PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
JSON_MEDIA_TYPE = "application/json"
_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]*")


def valid_artifact_name(name: str) -> bool:
    """Whether ``name`` can be the last part of a reference and a file name."""
    return _NAME.fullmatch(name) is not None and not name.endswith(".")


class ArtifactRef(Contract):
    """Where a check's bulky data is and what it is; the data is not in the run."""

    ref: str = Field(
        min_length=1,
        description="`<check_id>/<name>`, unique within a run; "
        "`Run.artifact(ref)` returns the payload.",
    )
    media_type: str = Field(
        description="`application/vnd.apache.parquet` for a table, "
        "`application/json` for a JSON value."
    )
    rows: int | None = Field(
        default=None, ge=0, description="Rows of a table; null for a JSON value."
    )
    bytes: int = Field(ge=0, description="Size of the file the artifact is stored as.")
    description: str = ""

    @field_validator("ref")
    @classmethod
    def _check_ref(cls, value: str) -> str:
        check_id, _, name = value.partition("/")
        if not check_id or not valid_artifact_name(name):
            raise ValueError(f"{value!r} is not <check_id>/<name>")
        return value


def payload_extension(media_type: str) -> str:
    """The file extension an artifact of ``media_type`` is stored under."""
    return "parquet" if media_type == PARQUET_MEDIA_TYPE else "json"


Payload = Any  # a pyarrow.Table, or a JSON-safe value
