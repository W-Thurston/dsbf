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
    PlanManifest,
    PlannedCheck,
    SourceKind,
    SourceRef,
)
from dsbf.model.params import CheckParams
from dsbf.model.policy import (
    ActionKind,
    ConfidenceComparison,
    Matcher,
    MetricComparison,
    Operator,
    Policy,
    PolicyAction,
    PolicyOutcome,
    Rule,
    RuleOutcome,
    SeverityComparison,
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
    Missingness,
    NullPair,
    NumericProfile,
    NumericStats,
    PatternCount,
    Quantile,
    SemanticType,
    TextProfile,
    TextShape,
    UnknownProfile,
    ValueCount,
)
from dsbf.model.run import SCHEMA_VERSION, Run, order_findings
from dsbf.model.subject import Subject, SubjectKind
from dsbf.model.timing import Timing
from dsbf.model.vocabulary import (
    CheckStatus,
    Confidence,
    Cost,
    Dimension,
    Severity,
    Stage,
)

__all__ = [
    "Action",
    "ActionKind",
    "AssumptionCheck",
    "BooleanProfile",
    "CategoricalProfile",
    "CheckParams",
    "CheckResult",
    "CheckStatus",
    "ColumnCounts",
    "ColumnProfile",
    "Confidence",
    "ConfidenceComparison",
    "Contract",
    "Cost",
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
    "Missingness",
    "NullPair",
    "Matcher",
    "Metric",
    "MetricComparison",
    "Note",
    "NumericProfile",
    "NumericStats",
    "Operator",
    "order_findings",
    "PatternCount",
    "PlanManifest",
    "PlannedCheck",
    "Policy",
    "PolicyAction",
    "PolicyOutcome",
    "Quantile",
    "Run",
    "Rule",
    "RuleOutcome",
    "RunHealth",
    "SamplingInfo",
    "Scalar",
    "SCHEMA_VERSION",
    "SemanticType",
    "Severity",
    "Stage",
    "SeverityComparison",
    "SourceKind",
    "SourceRef",
    "StatusCounts",
    "Subject",
    "SubjectKind",
    "TextProfile",
    "TextShape",
    "Timing",
    "UnknownProfile",
    "ValueCount",
]
