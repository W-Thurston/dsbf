"""Time a full run on each scenario and compare it with the committed baseline.

Run with ``uv run pytest -m perf``; see ``tests/perf/README.md``. Each run writes a
results JSON (``--perf-results``), and fails when a stage, task, the total, the peak
memory or a dashboard payload regressed against ``baseline.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.perf.compare import TOLERANCE, find_regressions
from tests.perf.harness import BASELINE_PATH, load_results, measure
from tests.perf.scenarios import SCENARIOS, Scenario

pytestmark = pytest.mark.perf


def _param(scenario: Scenario):
    marks = [pytest.mark.slow] if scenario.name.startswith("kitchen") else []
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
