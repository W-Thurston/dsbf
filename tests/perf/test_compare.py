from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.perf.compare import (
    ENGINE_BUDGETS,
    PROFILE_BUDGETS,
    check_budget,
    check_engine_budget,
    find_regressions,
    render_summary,
)
from tests.perf.harness import BASELINE_PATH, load_results
from tests.perf.scenarios import (
    ENGINE_SCENARIOS,
    PER_CORE,
    PROFILE_SCENARIOS,
)

pytestmark = pytest.mark.unit


def entry(**overrides: Any) -> dict[str, Any]:
    base = {
        "total_seconds": 100.0,
        "peak_rss_mb": 1000.0,
        "stages": {"graph.run (all tasks)": 90.0, "persist_run": 0.2},
        "tasks": {"slow_task": 40.0, "quick_task": 0.2},
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
    )
    kinds = {r.kind for r in find_regressions("s", entry(), current)}
    assert kinds == {"total", "peak memory", "stage"}


def test_worst_regression_comes_first() -> None:
    current = entry(total_seconds=130.0, peak_rss_mb=2000.0)
    kinds = [r.kind for r in find_regressions("s", entry(), current)]
    assert kinds == ["peak memory", "total"]


def test_values_on_one_side_only_are_skipped() -> None:
    current = entry(tasks={"new_task": 500.0})
    assert find_regressions("s", entry(), current) == []


def test_summary_marks_a_scenario_without_a_baseline() -> None:
    results = {"scenarios": {"a": entry(), "b": copy.deepcopy(entry())}}
    baseline = {"scenarios": {"a": entry(total_seconds=50.0)}}
    rows = render_summary(results, baseline).splitlines()
    assert "+100%" in rows[2] and "`slow_task` 40.0" in rows[2]
    assert "none" in rows[3]


def profile_entry(**overrides: Any) -> dict[str, Any]:
    base = {
        "total_seconds": 0.8,
        "peak_rss_mb": 300.0,
        "stages": {"load": 1.0, "infer_column_types": 0.3, "profile_dataset": 0.5},
        "memory_ratio": 1.1,
        "peak_isolated": True,
    }
    return {**base, **overrides}


def test_profile_entry_within_budget_breaks_nothing() -> None:
    assert check_budget("profile-kitchen50k", profile_entry()) == []


def test_profile_pass_over_its_time_budget_is_reported() -> None:
    [broken] = check_budget("profile-kitchen50k", profile_entry(total_seconds=2.4))
    assert "2.40 s" in broken and "budget 2.0 s" in broken


def test_profile_pass_over_its_memory_budget_is_reported() -> None:
    [broken] = check_budget("profile-kitchen50k", profile_entry(memory_ratio=2.8))
    assert "2.80x" in broken and "budget 2.5x" in broken


def test_time_budget_is_unset_for_the_wide_scenario() -> None:
    assert check_budget("profile-wide1m", profile_entry(total_seconds=40.0)) == []


def test_memory_that_includes_the_load_is_not_held_to_the_budget() -> None:
    assert (
        check_budget(
            "profile-kitchen50k", profile_entry(memory_ratio=9.0, peak_isolated=False)
        )
        == []
    )


def test_scenario_without_a_budget_breaks_nothing() -> None:
    assert check_budget("titanic", entry(total_seconds=1e6)) == []


def test_every_profile_scenario_has_a_budget() -> None:
    assert {s.name for s in PROFILE_SCENARIOS} == set(PROFILE_BUDGETS)


def test_summary_lists_stages_for_an_entry_without_tasks() -> None:
    rows = render_summary({"scenarios": {"p": profile_entry()}}, {}).splitlines()
    assert "`load` 1.0" in rows[2] and "`profile_dataset` 0.5" in rows[2]


def test_profile_entry_is_compared_without_tasks() -> None:
    slower = profile_entry(
        stages={"load": 1.0, "infer_column_types": 0.3, "profile_dataset": 2.0}
    )
    [found] = find_regressions("p", profile_entry(), slower)
    assert (found.kind, found.name) == ("stage", "profile_dataset")


def test_the_engine_scenarios_pair_each_dataset_serial_and_parallel() -> None:
    by_data = {}
    for scenario in ENGINE_SCENARIOS:
        if scenario.name.startswith("engine-"):
            by_data.setdefault(scenario.name.split("-")[1], set()).add(scenario.workers)
    assert by_data == {"titanic": {1, PER_CORE}, "kitchen5k": {1, PER_CORE}}


def test_every_standard_scenario_has_a_budget_and_runs_the_default_plan() -> None:
    standard = [s for s in ENGINE_SCENARIOS if s.name.startswith("standard-")]
    assert {s.name for s in standard} == set(ENGINE_BUDGETS)
    assert {(s.depth, s.workers) for s in standard} == {("standard", None)}


def test_an_engine_entry_within_its_budget_breaks_nothing() -> None:
    assert check_engine_budget("standard-titanic", entry(total_seconds=4.0)) == []


def test_a_broken_budget_is_described() -> None:
    missed = check_engine_budget(
        "standard-kitchen50k", entry(total_seconds=90.0, peak_rss_mb=1500.0)
    )
    assert missed == [
        "standard-kitchen50k: took 90.0 s, budget 60 s",
        "standard-kitchen50k: peak memory 1,500 MB, budget 1,024 MB",
    ]


def test_the_summary_reports_the_budgets() -> None:
    results = {
        "scenarios": {
            "standard-titanic": entry(total_seconds=30.0),
            "standard-kitchen50k": entry(total_seconds=50.0, peak_rss_mb=900.0),
        }
    }
    summary = render_summary(results, {})
    assert "- standard-titanic: took 30.0 s, budget 5 s" in summary
    assert "- standard-kitchen50k: within its budget" in summary


def test_committed_baseline_covers_every_scenario() -> None:
    # Recorded on the CI runner: copy a perf run's perf.json (README.md).
    baseline = load_results(BASELINE_PATH)["scenarios"]
    missing = [
        s.name
        for s in [*PROFILE_SCENARIOS, *ENGINE_SCENARIOS]
        if s.name not in baseline
    ]
    assert not missing, f"record a baseline for {missing} from the CI runner"
