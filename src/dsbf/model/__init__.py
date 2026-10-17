# dsbf/model/__init__.py
"""The result contracts: pydantic models with no dependency on the engine.

Importing this package pulls in pydantic and nothing else, so embedded users and
the schema export don't pay for the analysis libraries.
"""

from dsbf.model.base import Contract, FiniteFloat
from dsbf.model.evidence import (
    AssumptionCheck,
    Evidence,
    Interval,
    Metric,
    SamplingInfo,
)
from dsbf.model.subject import Subject, SubjectKind
from dsbf.model.timing import Timing
from dsbf.model.vocabulary import Confidence, Dimension, Severity

__all__ = [
    "AssumptionCheck",
    "Confidence",
    "Contract",
    "Dimension",
    "Evidence",
    "FiniteFloat",
    "Interval",
    "Metric",
    "SamplingInfo",
    "Severity",
    "Subject",
    "SubjectKind",
    "Timing",
]
