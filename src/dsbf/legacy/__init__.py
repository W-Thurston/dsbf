# dsbf/legacy/__init__.py
"""Run the legacy ``BaseTask`` set on the new engine, until each task is migrated."""

from dsbf.legacy.adapter import (
    EXCLUDED_TASKS,
    LEGACY_PACK,
    UNDECLARED_READS,
    LegacyParams,
    adapt_task,
    legacy_checks,
)
from dsbf.legacy.context import LegacyContext
from dsbf.legacy.result import LegacyTaskOutput
from dsbf.legacy.session import LegacySession

__all__ = [
    "EXCLUDED_TASKS",
    "LEGACY_PACK",
    "UNDECLARED_READS",
    "LegacyContext",
    "LegacyParams",
    "LegacySession",
    "LegacyTaskOutput",
    "adapt_task",
    "legacy_checks",
]
