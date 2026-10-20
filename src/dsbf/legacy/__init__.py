# dsbf/legacy/__init__.py
"""Run the legacy ``BaseTask`` set on the new engine, until each task is migrated."""

from dsbf.legacy.adapter import (
    EXCLUDED_TASKS,
    LEGACY_PACK,
    SUPERSEDED_PACK,
    UNDECLARED_READS,
    LegacyParams,
    adapt_task,
    legacy_checks,
)
from dsbf.legacy.context import LegacyContext
from dsbf.legacy.result import LegacyTaskOutput
from dsbf.legacy.session import LegacySession
from dsbf.legacy.supersedes import SUPERSEDED_BY, make_legacy_plan, superseded_tasks

__all__ = [
    "EXCLUDED_TASKS",
    "LEGACY_PACK",
    "SUPERSEDED_BY",
    "SUPERSEDED_PACK",
    "UNDECLARED_READS",
    "LegacyContext",
    "LegacyParams",
    "LegacySession",
    "LegacyTaskOutput",
    "adapt_task",
    "legacy_checks",
    "make_legacy_plan",
    "superseded_tasks",
]
