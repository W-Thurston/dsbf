# dsbf/model/vocabulary.py
"""Closed vocabularies shared by findings, policies and the dashboard."""

from enum import StrEnum
from typing import Any, Self


class _Ordinal(StrEnum):
    """A string enum whose member order is its rank, lowest first.

    A plain ``StrEnum`` would order alphabetically ("error" before "info"), so the
    comparisons are overridden on purpose.
    """

    def _rank(self) -> int:
        return list(type(self)).index(self)

    def _compare(self, other: Any, op: str) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return getattr(self._rank(), op)(other._rank())

    def __lt__(self, other: Self) -> bool:  # type: ignore[override]
        return self._compare(other, "__lt__")

    def __le__(self, other: Self) -> bool:  # type: ignore[override]
        return self._compare(other, "__le__")

    def __gt__(self, other: Self) -> bool:  # type: ignore[override]
        return self._compare(other, "__gt__")

    def __ge__(self, other: Self) -> bool:  # type: ignore[override]
        return self._compare(other, "__ge__")


class Severity(_Ordinal):
    """How much a finding matters, from informational to blocking."""

    INFO = "info"
    NOTICE = "notice"
    WARN = "warn"
    ERROR = "error"


class Confidence(_Ordinal):
    """How firmly a check stands behind a finding; each check caps its own."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class Dimension(StrEnum):
    """The aspect of the data a finding speaks to.

    The first five are the quality dimensions; the rest describe the shape of
    the data.
    """

    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    USABILITY = "usability"
    REDUNDANCY = "redundancy"
    LEAKAGE = "leakage"
    DISTRIBUTION = "distribution"
    RELATIONSHIP = "relationship"
    TEMPORAL = "temporal"


class Stage(StrEnum):
    """How prepared the data is, and the stages a check can apply to.

    Inferring the stage of a dataset gives `raw`, `cleaned` or `modeling`. A check
    also declares `report` (it summarises rather than inspects) or `any`.
    """

    RAW = "raw"
    CLEANED = "cleaned"
    MODELING = "modeling"
    REPORT = "report"
    ANY = "any"


class CheckStatus(StrEnum):
    """How one check's execution ended. Every status but `ok` comes with a note."""

    OK = "ok"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"
