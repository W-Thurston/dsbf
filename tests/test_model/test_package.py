# tests/test_model/test_package.py
"""The contracts package stays light and exports to JSON Schema."""

import subprocess
import sys

import pytest

import dsbf.model as model

pytestmark = pytest.mark.unit

HEAVY = ("pandas", "numpy", "polars", "scipy", "sklearn", "pyarrow")


def test_importing_the_contracts_pulls_in_no_analysis_library():
    code = (
        "import sys, dsbf.model; "
        f"loaded = [m for m in {HEAVY!r} if m in sys.modules]; "
        "print(','.join(loaded))"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == ""


@pytest.mark.parametrize(
    "name",
    [
        "Subject",
        "Evidence",
        "Interval",
        "AssumptionCheck",
        "SamplingInfo",
        "Timing",
        "Action",
        "Guidance",
        "Finding",
        "Note",
        "CheckParams",
        "CheckResult",
        "StatusCounts",
        "HealthIssue",
        "RunHealth",
        "ColumnCounts",
        "Quantile",
        "NumericStats",
        "Histogram",
        "ValueCount",
        "LengthStats",
        "NumericProfile",
        "CategoricalProfile",
        "BooleanProfile",
        "DatetimeProfile",
        "TextProfile",
        "IdentifierProfile",
        "UnknownProfile",
        "DatasetProfile",
        "SourceRef",
        "DatasetManifest",
        "PlannedCheck",
        "PlanManifest",
        "Run",
        "Matcher",
        "PolicyAction",
        "Rule",
        "Policy",
        "RuleOutcome",
        "PolicyOutcome",
        "SeverityComparison",
        "ConfidenceComparison",
        "MetricComparison",
    ],
)
def test_each_model_exports_json_schema(name):
    schema = getattr(model, name).model_json_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False


def test_the_vocabularies_export_as_enums():
    from pydantic import TypeAdapter

    for name, expected in [
        ("Severity", 4),
        ("Confidence", 3),
        ("Dimension", 8),
        ("CheckStatus", 6),
        ("ActionKind", 4),
        ("Operator", 6),
    ]:
        schema = TypeAdapter(getattr(model, name)).json_schema()
        assert len(schema["enum"]) == expected


def test_the_column_profile_union_exports_a_discriminator():
    from pydantic import TypeAdapter

    schema = TypeAdapter(model.ColumnProfile).json_schema()

    assert schema["discriminator"]["propertyName"] == "semantic_type"
    assert len(schema["oneOf"]) == 7
