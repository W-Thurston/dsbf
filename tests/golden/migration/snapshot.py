"""The core pack's findings on one dataset, reduced to what a reviewer judges.

A finding is kept as its id, check, code, subject, severity and evidence metrics, with
floats rounded as in the legacy snapshots. Messages, titles, timings and run ids are
left out: they change without the behaviour changing. A report-only check, which has
no findings, keeps the part of its output a reviewer judges: the levels, the coverage
and the columns each dimension flags.
"""

from __future__ import annotations

from typing import Any

from tests.golden.normalise import canonical

CORE_SNAPSHOT_FORMAT = 2


def _quality_dimensions(output: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "assessed": d["assessed"],
            "level": d["level"],
            "coverage": d["coverage"],
            "affected_columns": sorted(d["affected_columns"]),
            "findings": len(d["findings"]),
        }
        for name, d in output["dimensions"].items()
    }


def _ml_readiness(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "gate": output["gate"],
        "complete": output["complete"],
        "dimensions": {
            name: {
                "assessed": d["assessed"],
                "level": d["level"],
                "coverage": d["coverage"],
                "action_columns": sorted(d["action_columns"]),
                "findings": len(d["findings"]),
            }
            for name, d in output["dimensions"].items()
        },
        "clean_columns": sorted(output["clean_columns"]),
    }


# The checks that report on others, each to the summary of its output.
SUMMARISED_OUTPUTS = {
    "quality_dimensions": _quality_dimensions,
    "ml_readiness": _ml_readiness,
}


def core_snapshot(dataset: str, run: dict[str, Any]) -> dict[str, Any]:
    """The snapshot of a core ``Run`` (as a dict) on ``dataset``."""
    checks: dict[str, Any] = {}
    for check_id, result in sorted(run["results"].items()):
        findings = [
            {
                "id": finding["id"],
                "check": finding["check_id"],
                "code": finding["code"],
                "subject": {
                    "kind": finding["subject"]["kind"],
                    "columns": finding["subject"]["columns"],
                },
                "severity": finding["severity"],
                "metrics": canonical(finding["evidence"]["metrics"]),
            }
            for finding in result["findings"]
        ]
        findings.sort(key=lambda f: (f["subject"]["columns"], f["code"], f["id"]))
        checks[check_id] = {
            "version": result["check_version"],
            "status": result["status"],
            "findings": findings,
        }
        summarise = SUMMARISED_OUTPUTS.get(check_id)
        if summarise is not None and result["status"] == "ok":
            checks[check_id]["output"] = canonical(summarise(result["output"]))
    return {
        "format": CORE_SNAPSHOT_FORMAT,
        "dataset": dataset,
        "shape": [run["dataset"]["n_rows"], run["dataset"]["n_columns"]],
        "checks": checks,
    }
