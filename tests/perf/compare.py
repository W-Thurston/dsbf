"""Decide whether a run got slower, larger or hungrier than the committed baseline.

A value regresses when it grows by more than ``TOLERANCE`` *and* by at least a floor
for its kind. The relative limit alone would flag a task that went from 0.2 s to 0.3 s,
which on a shared CI runner is noise.

The profile pass has budgets as well: absolute limits that hold with or without a
baseline (``PROFILE_BUDGETS``). So does the standard plan (``ENGINE_BUDGETS``, the
Phase 4 exit criteria).
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
}
UNITS = {
    "total": "s",
    "stage": "s",
    "task": "s",
    "peak memory": "MB",
}


@dataclass(frozen=True)
class Budget:
    """Absolute limits for one profile scenario; ``None`` leaves a limit unset."""

    seconds: float | None
    memory_ratio: float | None


# The pass is type inference, the dataset profile and stage inference, without loading.
# ``memory_ratio`` is the loaded frame plus what the pass adds, over the frame.
PROFILE_BUDGETS = {
    "profile-kitchen50k": Budget(seconds=2.0, memory_ratio=2.5),
    "profile-wide1m": Budget(seconds=None, memory_ratio=2.5),
}


@dataclass(frozen=True)
class EngineBudget:
    """Absolute limits for one ``dsbf.profile()`` scenario; ``None`` leaves one off."""

    seconds: float
    peak_mb: float | None = None


# The standard plan's budgets (REFACTOR_PLAN section 11), measured on the CI runner.
ENGINE_BUDGETS = {
    "standard-titanic": EngineBudget(seconds=5.0),
    "standard-kitchen50k": EngineBudget(seconds=60.0, peak_mb=1024.0),
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
    return sorted(found, key=lambda r: r.growth, reverse=True)


def check_budget(scenario: str, entry: dict[str, Any]) -> list[str]:
    """Describe each absolute budget that ``entry`` breaks; empty if it keeps all."""
    budget = PROFILE_BUDGETS.get(scenario)
    if budget is None:
        return []
    broken = []
    seconds = entry["total_seconds"]
    if budget.seconds is not None and seconds > budget.seconds:
        broken.append(
            f"{scenario}: profile pass took {seconds:,.2f} s, "
            f"budget {budget.seconds:,.1f} s"
        )
    # Without an isolated peak the figure includes the load, so it isn't held to this.
    ratio = entry.get("memory_ratio")
    if (
        budget.memory_ratio is not None
        and ratio is not None
        and entry.get("peak_isolated", False)
        and ratio > budget.memory_ratio
    ):
        broken.append(
            f"{scenario}: peak memory {ratio:.2f}x the loaded data, "
            f"budget {budget.memory_ratio:.1f}x"
        )
    return broken


def check_engine_budget(scenario: str, entry: dict[str, Any]) -> list[str]:
    """Describe each budget in ``ENGINE_BUDGETS`` that ``entry`` breaks."""
    budget = ENGINE_BUDGETS.get(scenario)
    if budget is None:
        return []
    missed = []
    seconds = entry["total_seconds"]
    if seconds > budget.seconds:
        missed.append(
            f"{scenario}: took {seconds:,.1f} s, budget {budget.seconds:,.0f} s"
        )
    peak = entry["peak_rss_mb"]
    if budget.peak_mb is not None and peak > budget.peak_mb:
        missed.append(
            f"{scenario}: peak memory {peak:,.0f} MB, budget {budget.peak_mb:,.0f} MB"
        )
    return missed


def render_summary(
    results: dict[str, Any], baseline: dict[str, Any], top_tasks: int = 5
) -> str:
    """A Markdown table of the run against the baseline, for the CI job summary."""
    lines = [
        "| Scenario | Total (s) | Baseline (s) | Change | Peak RSS (MB) "
        "| Slowest tasks or stages (s) |",
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
        # A profile entry has stages and no tasks.
        parts = entry.get("tasks") or entry.get("stages", {})
        slowest = sorted(parts.items(), key=lambda kv: -kv[1])[:top_tasks]
        tasks = ", ".join(f"`{task}` {seconds:,.1f}" for task, seconds in slowest)
        lines.append(
            f"| {name} | {total:,.1f} | {base_cell} | {change} "
            f"| {entry['peak_rss_mb']:,.0f} | {tasks} |"
        )
    budgets = [
        line
        for name, entry in results["scenarios"].items()
        if name in ENGINE_BUDGETS
        for line in check_engine_budget(name, entry) or [f"{name}: within its budget"]
    ]
    if budgets:
        lines += ["", "Phase 4 budgets:", ""]
        lines += [f"- {line}" for line in budgets]
    return "\n".join(lines) + "\n"
