# dsbf/model/__init__.py
"""The result contracts: pydantic models with no dependency on the engine.

Importing this package pulls in pydantic and nothing else, so embedded users and
the schema export don't pay for the analysis libraries.
"""

from dsbf.model.base import Contract, FiniteFloat, Scalar
from dsbf.model.check import CheckResult, Note
from dsbf.model.evidence import (
    AssumptionCheck,
    Evidence,
    Interval,
    Metric,
    SamplingInfo,
)
from dsbf.model.finding import Action, Finding, Guidance, GuidancePhase
from dsbf.model.health import HealthIssue, RunHealth, StatusCounts
from dsbf.model.subject import Subject, SubjectKind
from dsbf.model.timing import Timing
from dsbf.model.vocabulary import CheckStatus, Confidence, Dimension, Severity

__all__ = [
    "Action",
    "AssumptionCheck",
    "CheckResult",
    "CheckStatus",
    "Confidence",
    "Contract",
    "Dimension",
    "Evidence",
    "FiniteFloat",
    "Finding",
    "Guidance",
    "GuidancePhase",
    "HealthIssue",
    "Interval",
    "Metric",
    "Note",
    "RunHealth",
    "SamplingInfo",
    "Scalar",
    "Severity",
    "StatusCounts",
    "Subject",
    "SubjectKind",
    "Timing",
]
