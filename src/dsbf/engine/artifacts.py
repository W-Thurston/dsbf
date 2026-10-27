# dsbf/engine/artifacts.py
"""Encode and decode the payloads checks attach: Arrow tables and JSON values."""

import io
import json
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dsbf.model.artifact import (
    JSON_MEDIA_TYPE,
    PARQUET_MEDIA_TYPE,
    ArtifactRef,
    Payload,
)
from dsbf.model.scrub import json_safe


def encode(payload: Payload) -> tuple[bytes, str, int | None, Payload]:
    """The file bytes, media type, row count and normalised payload of ``payload``.

    A ``pyarrow.Table`` is written as Parquet; anything else must be a JSON-safe value
    (non-finite numbers become null).

    Raises:
        ValueError: ``payload`` is neither.
    """
    if isinstance(payload, pa.Table):
        sink = io.BytesIO()
        pq.write_table(payload, sink, compression="zstd")
        return sink.getvalue(), PARQUET_MEDIA_TYPE, payload.num_rows, payload
    safe, _ = json_safe(payload)
    data = json.dumps(safe, separators=(",", ":"), ensure_ascii=False).encode()
    return data, JSON_MEDIA_TYPE, None, safe


def decode(data: bytes, media_type: str) -> Payload:
    """The payload stored as ``data``."""
    if media_type == PARQUET_MEDIA_TYPE:
        return pq.read_table(io.BytesIO(data))
    return json.loads(data)


def make_ref(
    ref: str, payload: Payload, description: str = ""
) -> tuple[ArtifactRef, Payload]:
    """The reference for ``payload`` and the payload as the run will hold it."""
    data, media_type, rows, normal = encode(payload)
    return (
        ArtifactRef(
            ref=ref,
            media_type=media_type,
            rows=rows,
            bytes=len(data),
            description=description,
        ),
        normal,
    )


def same(left: Any, right: Any) -> bool:
    """Whether two payloads hold the same data."""
    if isinstance(left, pa.Table) and isinstance(right, pa.Table):
        return left.equals(right)
    return left == right
