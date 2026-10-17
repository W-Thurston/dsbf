# tests/test_model/builders.py
"""Small constructors and helpers shared by the contract tests."""

import json
from typing import Any

from dsbf.model import (
    CheckResult,
    CheckStatus,
    Confidence,
    Dimension,
    Evidence,
    Finding,
    Note,
    Severity,
    Subject,
    Timing,
)


def strict_loads(text):
    """Parse JSON the way a strict consumer would: NaN and Infinity are errors."""

    def refuse(token):
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(text, parse_constant=refuse)


def round_trip(model):
    return type(model).model_validate_json(model.model_dump_json())


def make_evidence(**overrides):
    fields = {"metrics": {"null_rate": 0.4}, "n_used": 10, "n_total": 10}
    return Evidence.model_validate({"method": "null_rate", **fields, **overrides})


def make_finding(**overrides):
    fields: dict[str, Any] = {
        "check_id": "summarize_nulls",
        "check_version": "1",
        "code": "high_null_rate",
        "subject": Subject.column("age"),
        "dimension": Dimension.COMPLETENESS,
        "severity": Severity.WARN,
        "confidence": Confidence.HIGH,
        "title": "High null rate",
        "message": "'age' is 40% null.",
        "evidence": make_evidence(),
    }
    return Finding.build(**{**fields, **overrides})


def make_note(**overrides):
    fields = {"code": "dependency_failed", "message": "Requires 'skewness' (error)."}
    return Note(**{**fields, **overrides})


def make_result(status=CheckStatus.OK, **overrides):
    fields: dict[str, Any] = {
        "check_id": "summarize_nulls",
        "check_version": "1",
        "status": status,
        "timing": Timing(wall_seconds=0.5),
    }
    check_id = overrides.get("check_id", fields["check_id"])
    if status == CheckStatus.OK:
        fields["findings"] = (make_finding(check_id=check_id),)
    else:
        fields["notes"] = (make_note(),)
    return CheckResult(**{**fields, **overrides})
