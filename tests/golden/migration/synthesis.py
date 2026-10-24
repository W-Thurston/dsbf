"""What differs between the synthesis checks and the scorers they replace.

``data_quality_scorer`` and ``ml_readiness_scorer`` read the legacy tasks' results;
``quality_dimensions`` and ``ml_readiness`` read the core checks' findings. They agree
when each dimension has the same level and flags the same columns, and the readiness
gate is the same.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.golden.migration.harness import Difference, Result

SHOWN = 2


def _names(columns: list[str]) -> str:
    """The count, then the first columns: enough to see the list change."""
    shown = ", ".join(columns[:SHOWN]) + (", ..." if len(columns) > SHOWN else "")
    return f"{len(columns)} columns: {shown}"


def _dimension(
    name: str, was: Mapping[str, Any], now: Mapping[str, Any], columns: str
) -> list[Difference]:
    found: list[Difference] = []
    after = now["level"] if now["assessed"] else "not assessed"
    if was["level"] != after:
        found.append(Difference(name, "level", f"legacy {was['level']}, core {after}"))
    old, new = set(was["affected_columns"]), set(now[columns])
    if old - new:
        found.append(Difference(name, "legacy_only", _names(sorted(old - new))))
    if new - old:
        found.append(Difference(name, "core_only", _names(sorted(new - old))))
    return found


def compare_data_quality_scorer(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]["categories"]
    new = core["output"]["dimensions"]
    return [
        d
        for name in sorted(old)
        for d in _dimension(name, old[name], new[name], "affected_columns")
    ]


def compare_ml_readiness_scorer(legacy: Result, core: Result) -> list[Difference]:
    data = legacy["result"]["data"]
    old = data["categories"]
    new = core["output"]["dimensions"]
    found = [
        d
        for name in sorted(old)
        for d in _dimension(name, old[name], new[name], "action_columns")
    ]
    gate = core["output"]["gate"]
    if data["readiness_gate"] != gate:
        found.append(
            Difference("", "gate", f"legacy {data['readiness_gate']}, core {gate}")
        )
    return found
