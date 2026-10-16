from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.perf.compare import find_regressions, render_summary
from tests.perf.harness import BASELINE_PATH, load_results
from tests.perf.scenarios import SCENARIOS

pytestmark = pytest.mark.unit


def entry(**overrides: Any) -> dict[str, Any]:
    base = {
        "total_seconds": 100.0,
        "peak_rss_mb": 1000.0,
        "stages": {"graph.run (all tasks)": 90.0, "persist_run": 0.2},
        "tasks": {"slow_task": 40.0, "quick_task": 0.2},
        "api": {"run": {"status": 200, "bytes": 200_000, "milliseconds": 5.0}},
    }
    return {**base, **overrides}


def test_identical_runs_have_no_regressions() -> None:
    assert find_regressions("s", entry(), entry()) == []


def test_growth_beyond_tolerance_and_floor_is_reported() -> None:
    current = entry(tasks={"slow_task": 51.0, "quick_task": 0.2})
    [found] = find_regressions("s", entry(), current)
    assert (found.kind, found.name) == ("task", "slow_task")
    assert found.growth == pytest.approx(0.275)
    assert "slow_task" in found.describe()


def test_growth_within_tolerance_is_not_reported() -> None:
    current = entry(tasks={"slow_task": 49.0, "quick_task": 0.2})
    assert find_regressions("s", entry(), current) == []


def test_small_absolute_growth_is_noise() -> None:
    """A task that doubled from 0.2 s to 0.5 s is below the one-second floor."""
    current = entry(tasks={"slow_task": 40.0, "quick_task": 0.5})
    assert find_regressions("s", entry(), current) == []


def test_every_gated_kind_is_checked() -> None:
    current = entry(
        total_seconds=140.0,
        peak_rss_mb=1400.0,
        stages={"graph.run (all tasks)": 120.0, "persist_run": 0.2},
        api={"run": {"status": 200, "bytes": 400_000, "milliseconds": 5.0}},
    )
    kinds = {r.kind for r in find_regressions("s", entry(), current)}
    assert kinds == {"total", "peak memory", "stage", "api payload"}


def test_worst_regression_comes_first() -> None:
    current = entry(total_seconds=130.0, peak_rss_mb=2000.0)
    kinds = [r.kind for r in find_regressions("s", entry(), current)]
    assert kinds == ["peak memory", "total"]


def test_values_on_one_side_only_are_skipped() -> None:
    current = entry(
        tasks={"new_task": 500.0}, api={"new": {"status": 200, "bytes": 10**9}}
    )
    assert find_regressions("s", entry(), current) == []


def test_endpoint_that_stops_answering_is_reported() -> None:
    broken = {"run": {"status": 500, "bytes": 21, "milliseconds": 5.0}}
    [found] = find_regressions("s", entry(), entry(api=broken))
    assert found.kind == "api status"
    assert found.describe().endswith("HTTP 200 -> HTTP 500")


def test_endpoint_that_already_failed_is_not_reported() -> None:
    failing = {"run": {"status": 500, "bytes": 21, "milliseconds": 5.0}}
    assert find_regressions("s", entry(api=failing), entry(api=failing)) == []


def test_api_milliseconds_and_errors_are_not_gated() -> None:
    current = entry(api={"run": {"status": 200, "bytes": 200_000, "milliseconds": 9e5}})
    assert find_regressions("s", entry(), current) == []
    assert find_regressions("s", entry(), entry(api={"error": "boom"})) == []


def test_summary_marks_a_scenario_without_a_baseline() -> None:
    results = {"scenarios": {"a": entry(), "b": copy.deepcopy(entry())}}
    baseline = {"scenarios": {"a": entry(total_seconds=50.0)}}
    rows = render_summary(results, baseline).splitlines()
    assert "+100%" in rows[2] and "`slow_task` 40.0" in rows[2]
    assert "none" in rows[3]


def test_committed_baseline_covers_every_scenario() -> None:
    baseline = load_results(BASELINE_PATH)["scenarios"]
    missing = [s.name for s in SCENARIOS if s.name not in baseline]
    assert not missing, f"record a baseline for {missing} (--update-perf-baseline)"
