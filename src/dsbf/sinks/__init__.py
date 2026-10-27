# dsbf/sinks/__init__.py
"""Where a ``Run`` goes when it is written to disk, and how one is read back."""

from dsbf.sinks.files import RUN_FILE, read_artifact, read_run, write_run
from dsbf.sinks.parquet import PARQUET_SINKS
from dsbf.sinks.registry import (
    RunReadError,
    Sink,
    get_sinks,
    known_sinks,
    register_alias,
    register_sink,
)
from dsbf.sinks.report import REPORT_FILE, render_report

__all__ = [
    "PARQUET_SINKS",
    "REPORT_FILE",
    "RUN_FILE",
    "RunReadError",
    "Sink",
    "get_sinks",
    "known_sinks",
    "read_artifact",
    "read_run",
    "register_alias",
    "register_sink",
    "render_report",
    "write_run",
]
