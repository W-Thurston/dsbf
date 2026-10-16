"""Decide whether a run got slower, larger or hungrier than the committed baseline.

A value regresses when it grows by more than ``TOLERANCE`` *and* by at least a floor
for its kind. The relative limit alone would flag a task that went from 0.2 s to 0.3 s,
which on a shared CI runner is noise. A dashboard endpoint that stops answering 200
regresses regardless of size.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

TOLERANCE = 0.25
# Minimum absolute growth before a relative regression counts.
FLOORS = {
    "total": 1.0,
    "stage": 1.0,
    "task": 1.0,
    "peak memory": 100.0,
    "api payload": 10 * 1024.0,
}
UNITS = {
    "total": "s",
    "stage": "s",
    "task": "s",
    "peak memory": "MB",
    "api payload": "B",
    "api status": "",
}


@dataclass(frozen=True)
class Regression:
    scenario: str
    kind: str
    name: str
    baseline: float
    current: float

    @property
    def growth(self) -> float:
        return self.current / self.baseline - 1

    def describe(self) -> str:
        if self.kind == "api status":
            return (
                f"{self.scenario}: {self.kind} {self.name}: "
                f"HTTP {self.baseline:.0f} -> HTTP {self.current:.0f}"
            )
        unit = UNITS[self.kind]
        return (
            f"{self.scenario}: {self.kind} {self.name}: "
            f"{self.baseline:,.1f} {unit} -> {self.current:,.1f} {unit} "
            f"(+{self.growth:.0%})"
        )


def _measurements(entry: dict[str, Any]) -> Iterator[tuple[str, str, float]]:
    """Yield (kind, name, value) for every gated value in a scenario entry."""
    yield "total", "run", entry["total_seconds"]
    yield "peak memory", "process", entry["peak_rss_mb"]
    for name, seconds in entry.get("stages", {}).items():
        yield "stage", name, seconds
    for name, seconds in entry.get("tasks", {}).items():
        yield "task", name, seconds
    for name, probe in entry.get("api", {}).items():
        if isinstance(probe, dict) and "bytes" in probe:
            yield "api payload", name, float(probe["bytes"])


def find_regressions(
    scenario: str,
    baseline: dict[str, Any],
    current: dict[str, Any],
    tolerance: float = TOLERANCE,
) -> list[Regression]:
    """Return every value in ``current`` that regressed against ``baseline``.

    Values present on only one side are not compared: a new task has no baseline, and a
    removed one cannot regress.
    """
    before = {(k, n): v for k, n, v in _measurements(baseline)}
    found = []
    for kind, name, value in _measurements(current):
        reference = before.get((kind, name))
        if reference is None or reference <= 0:
            continue
        if value > reference * (1 + tolerance) and value - reference >= FLOORS[kind]:
            found.append(Regression(scenario, kind, name, reference, value))
    found += _status_regressions(scenario, baseline, current)
    return sorted(found, key=lambda r: r.growth, reverse=True)


def _status_regressions(
    scenario: str, baseline: dict[str, Any], current: dict[str, Any]
) -> list[Regression]:
    """Endpoints that answered 200 in the baseline and no longer do."""
    found = []
    for name, probe in current.get("api", {}).items():
        before = baseline.get("api", {}).get(name)
        if not (isinstance(probe, dict) and isinstance(before, dict)):
            continue
        if before.get("status") == 200 and probe.get("status") != 200:
            found.append(
                Regression(scenario, "api status", name, 200, probe.get("status", 0))
            )
    return found


def render_summary(
    results: dict[str, Any], baseline: dict[str, Any], top_tasks: int = 5
) -> str:
    """A Markdown table of the run against the baseline, for the CI job summary."""
    lines = [
        "| Scenario | Total (s) | Baseline (s) | Change | Peak RSS (MB) "
        "| Slowest tasks (s) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, entry in results["scenarios"].items():
        reference = baseline.get("scenarios", {}).get(name)
        total = entry["total_seconds"]
        if reference:
            before = reference["total_seconds"]
            base_cell, change = f"{before:,.1f}", f"{total / before - 1:+.0%}"
        else:
            base_cell, change = "none", ""
        slowest = sorted(entry["tasks"].items(), key=lambda kv: -kv[1])[:top_tasks]
        tasks = ", ".join(f"`{task}` {seconds:,.1f}" for task, seconds in slowest)
        lines.append(
            f"| {name} | {total:,.1f} | {base_cell} | {change} "
            f"| {entry['peak_rss_mb']:,.0f} | {tasks} |"
        )
    return "\n".join(lines) + "\n"
