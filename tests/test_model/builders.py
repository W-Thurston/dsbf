# tests/test_model/builders.py
"""Small constructors and helpers shared by the contract tests."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from dsbf.model import (
    CheckResult,
    CheckStatus,
    ColumnCounts,
    Confidence,
    DatasetManifest,
    DatasetProfile,
    Dimension,
    Evidence,
    Finding,
    Histogram,
    Note,
    NumericProfile,
    NumericStats,
    PlanManifest,
    PlannedCheck,
    Quantile,
    Run,
    Severity,
    SourceRef,
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


# ── profiles, manifests and runs ─────────────────────────────────────────────


def make_counts(rows=10, nulls=0, distinct=None, **overrides):
    fields = {
        "rows": rows,
        "nulls": nulls,
        "distinct": rows - nulls if distinct is None else distinct,
    }
    return ColumnCounts(**{**fields, **overrides})


def make_numeric(name="age", position=0, rows=10, **overrides):
    fields: dict[str, Any] = {
        "name": name,
        "position": position,
        "physical_dtype": "Float64",
        "counts": make_counts(rows=rows, nulls=1),
        "zeros": 0,
        "stats": NumericStats(
            min=1.0,
            max=9.0,
            mean=5.0,
            std=2.0,
            quantiles=(Quantile(p=0.25, value=3.0), Quantile(p=0.75, value=7.0)),
            quantile_method="exact",
        ),
        "histogram": Histogram(edges=(1.0, 5.0, 9.0), counts=(4, 4), rule="sturges"),
    }
    return NumericProfile(**{**fields, **overrides})


def make_dataset_profile(columns=None, n_rows=10, **overrides):
    columns = (make_numeric(rows=n_rows),) if columns is None else columns
    fields: dict[str, Any] = {"n_rows": n_rows, "columns": tuple(columns)}
    return DatasetProfile(**{**fields, **overrides})


def make_dataset_manifest(n_rows=10, n_columns=1, **overrides):
    fields: dict[str, Any] = {
        "name": "titanic",
        "source": SourceRef(kind="file", location="titanic.csv", format="csv"),
        "fingerprint": "sha256:0f3c",
        "n_rows": n_rows,
        "n_columns": n_columns,
    }
    return DatasetManifest(**{**fields, **overrides})


def make_plan(*check_ids, version="1", **overrides):
    fields: dict[str, Any] = {
        "dsbf_version": "0.1.0",
        "checks": tuple(
            PlannedCheck(check_id=c, check_version=version) for c in check_ids
        ),
    }
    return PlanManifest(**{**fields, **overrides})


STARTED = datetime(2026, 10, 17, 9, 0, tzinfo=UTC)


def make_run(results=None, **overrides):
    results = [make_result()] if results is None else list(results)
    fields: dict[str, Any] = {
        "run_id": UUID("7f3c1a52-4b0e-4d6a-9c1e-2a5b8d0f3e71"),
        "started_at": STARTED,
        "finished_at": STARTED + timedelta(seconds=3),
        "dataset": make_dataset_manifest(),
        "plan": make_plan(*(r.check_id for r in results)),
        "profile": make_dataset_profile(),
    }
    return Run.build(results=results, **{**fields, **overrides})
