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
from dsbf.model.manifest import (
    DatasetManifest,
    PackRef,
    PlanManifest,
    PlannedCheck,
    SourceKind,
    SourceRef,
)
from dsbf.model.profile import (
    BooleanProfile,
    CategoricalProfile,
    ColumnCounts,
    ColumnProfile,
    DatasetProfile,
    DatetimeProfile,
    Histogram,
    IdentifierProfile,
    LengthStats,
    NumericProfile,
    NumericStats,
    Quantile,
    SemanticType,
    TextProfile,
    UnknownProfile,
    ValueCount,
)
from dsbf.model.run import SCHEMA_VERSION, Run, order_findings
from dsbf.model.subject import Subject, SubjectKind
from dsbf.model.timing import Timing
from dsbf.model.vocabulary import CheckStatus, Confidence, Dimension, Severity

__all__ = [
    "Action",
    "AssumptionCheck",
    "BooleanProfile",
    "CategoricalProfile",
    "CheckResult",
    "CheckStatus",
    "ColumnCounts",
    "ColumnProfile",
    "Confidence",
    "Contract",
    "DatasetManifest",
    "DatasetProfile",
    "DatetimeProfile",
    "Dimension",
    "Evidence",
    "Finding",
    "FiniteFloat",
    "Guidance",
    "GuidancePhase",
    "HealthIssue",
    "Histogram",
    "IdentifierProfile",
    "Interval",
    "LengthStats",
    "Metric",
    "Note",
    "NumericProfile",
    "NumericStats",
    "order_findings",
    "PackRef",
    "PlanManifest",
    "PlannedCheck",
    "Quantile",
    "Run",
    "RunHealth",
    "SamplingInfo",
    "Scalar",
    "SCHEMA_VERSION",
    "SemanticType",
    "Severity",
    "SourceKind",
    "SourceRef",
    "StatusCounts",
    "Subject",
    "SubjectKind",
    "TextProfile",
    "Timing",
    "UnknownProfile",
    "ValueCount",
]
