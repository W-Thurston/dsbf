# dsbf/sinks/__init__.py
"""Where a ``Run`` goes when it is written to disk, and how one is read back."""

from dsbf.sinks.files import RUN_FILE, read_run, write_run
from dsbf.sinks.parquet import PARQUET_SINKS
from dsbf.sinks.registry import (
    RunReadError,
    Sink,
    get_sinks,
    known_sinks,
    register_alias,
    register_sink,
)

__all__ = [
    "PARQUET_SINKS",
    "RUN_FILE",
    "RunReadError",
    "Sink",
    "get_sinks",
    "known_sinks",
    "read_run",
    "register_alias",
    "register_sink",
    "write_run",
]
