"""The core pack's findings on one dataset, reduced to what a reviewer judges.

A finding is kept as its id, check, code, subject, severity and evidence metrics, with
floats rounded as in the legacy snapshots. Messages, titles, timings and run ids are
left out: they change without the behaviour changing.
"""

from __future__ import annotations

from typing import Any

from tests.golden.normalise import canonical

CORE_SNAPSHOT_FORMAT = 1


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
    return {
        "format": CORE_SNAPSHOT_FORMAT,
        "dataset": dataset,
        "shape": [run["dataset"]["n_rows"], run["dataset"]["n_columns"]],
        "checks": checks,
    }
