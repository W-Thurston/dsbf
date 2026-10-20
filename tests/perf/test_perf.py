"""Time a full run on each scenario and compare it with the committed baseline.

Run with ``uv run pytest -m perf``; see ``tests/perf/README.md``. Each run writes a
results JSON (``--perf-results``), and fails when a stage, task, the total, the peak
memory or a dashboard payload regressed against ``baseline.json``. The profile pass is
also held to the absolute budgets in ``compare.PROFILE_BUDGETS``; the standard plan's
targets in ``compare.ENGINE_BUDGETS`` are only reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.perf.compare import (
    TOLERANCE,
    check_budget,
    check_engine_budget,
    find_regressions,
)
from tests.perf.harness import BASELINE_PATH, load_results, measure
from tests.perf.scenarios import (
    ENGINE_SCENARIOS,
    PROFILE_SCENARIOS,
    SCENARIOS,
    EngineScenario,
    ProfileScenario,
    Scenario,
)

pytestmark = pytest.mark.perf


def _param(scenario: Scenario | ProfileScenario | EngineScenario):
    stem = scenario.name.split("-", 1)[-1] if "-" in scenario.name else scenario.name
    marks = [pytest.mark.slow] if stem.startswith(("kitchen", "wide")) else []
    return pytest.param(scenario, id=scenario.name, marks=marks)


@pytest.mark.parametrize("scenario", [_param(s) for s in SCENARIOS])
def test_run_within_baseline(
    scenario: Scenario,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    measured: dict[str, Any],
) -> None:
    config = request.config
    result = measure(scenario, tmp_path, cprofile=config.getoption("--perf-cprofile"))
    measured[scenario.name] = result

    # Timings from a run that crashed tasks describe less work than the baseline did.
    assert not result["failed_tasks"], f"tasks failed: {result['failed_tasks']}"
    assert "error" not in result["api"], result["api"]

    if config.getoption("--update-perf-baseline") or config.getoption(
        "--perf-no-compare"
    ):
        return

    baseline = load_results(BASELINE_PATH)["scenarios"].get(scenario.name)
    if baseline is None:
        pytest.skip(
            f"no baseline for {scenario.name}; record one with --update-perf-baseline"
        )
    regressions = find_regressions(scenario.name, baseline, result)
    assert not regressions, (
        f"regressed by more than {TOLERANCE:.0%} against tests/perf/baseline.json:\n"
        + "\n".join(f"  {r.describe()}" for r in regressions)
    )


@pytest.mark.parametrize("scenario", [_param(s) for s in PROFILE_SCENARIOS])
def test_profile_pass_within_budget(
    scenario: ProfileScenario,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    measured: dict[str, Any],
) -> None:
    config = request.config
    result = measure(scenario, tmp_path)
    measured[scenario.name] = result

    # The budget holds with or without a baseline, and when the baseline is being
    # rewritten: a pass that misses it should not become the new reference unnoticed.
    over_budget = check_budget(scenario.name, result)
    assert not over_budget, "over budget:\n" + "\n".join(f"  {b}" for b in over_budget)

    if config.getoption("--update-perf-baseline") or config.getoption(
        "--perf-no-compare"
    ):
        return

    baseline = load_results(BASELINE_PATH)["scenarios"].get(scenario.name)
    if baseline is None:
        pytest.skip(
            f"no baseline for {scenario.name}; record one with --update-perf-baseline"
        )
    regressions = find_regressions(scenario.name, baseline, result)
    assert not regressions, (
        f"regressed by more than {TOLERANCE:.0%} against tests/perf/baseline.json:\n"
        + "\n".join(f"  {r.describe()}" for r in regressions)
    )


@pytest.mark.parametrize("scenario", [_param(s) for s in ENGINE_SCENARIOS])
def test_engine_run_within_baseline(
    scenario: EngineScenario,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    measured: dict[str, Any],
) -> None:
    config = request.config
    result = measure(scenario, tmp_path)
    measured[scenario.name] = result

    # A check that errored or timed out did less work than the baseline's did.
    assert not result["failed_checks"], f"checks failed: {result['failed_checks']}"
    # Reported in the summary and here, never failed on: Phase 4 works towards them.
    for missed in check_engine_budget(scenario.name, result):
        print(f"target missed: {missed}")

    if config.getoption("--update-perf-baseline") or config.getoption(
        "--perf-no-compare"
    ):
        return

    baseline = load_results(BASELINE_PATH)["scenarios"].get(scenario.name)
    if baseline is None:
        pytest.skip(
            f"no baseline for {scenario.name}; record one with --update-perf-baseline"
        )
    regressions = find_regressions(scenario.name, baseline, result)
    assert not regressions, (
        f"regressed by more than {TOLERANCE:.0%} against tests/perf/baseline.json:\n"
        + "\n".join(f"  {r.describe()}" for r in regressions)
    )
