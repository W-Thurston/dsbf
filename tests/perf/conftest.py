from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.perf.compare import render_summary
from tests.perf.harness import (
    BASELINE_PATH,
    RESULTS_SCHEMA,
    describe_environment,
    load_results,
    write_results,
)


@pytest.fixture(scope="session")
def measured(pytestconfig: pytest.Config) -> Iterator[dict[str, Any]]:
    """Collect each scenario's measurements and write them when the session ends.

    The results file is written even if a scenario fails, so a regression can be
    inspected. ``--update-perf-baseline`` merges the measured scenarios into the
    baseline, keeping the entries of scenarios that were not run.
    """
    scenarios: dict[str, Any] = {}
    yield scenarios
    if not scenarios:
        return

    results = {
        "schema": RESULTS_SCHEMA,
        "environment": describe_environment(),
        "scenarios": scenarios,
    }
    # The invocation directory, not the temporary CWD the suite runs from.
    results_path = Path(pytestconfig.invocation_params.dir) / pytestconfig.getoption(
        "--perf-results"
    )
    write_results(results_path, results)
    summary = render_summary(results, load_results(BASELINE_PATH))
    results_path.with_suffix(".md").write_text(summary)

    if pytestconfig.getoption("--update-perf-baseline"):
        baseline = load_results(BASELINE_PATH)
        baseline["schema"] = RESULTS_SCHEMA
        baseline["environment"] = results["environment"]
        baseline["scenarios"] = {**baseline["scenarios"], **scenarios}
        write_results(BASELINE_PATH, baseline)
