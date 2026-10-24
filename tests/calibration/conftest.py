from __future__ import annotations

import platform
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from tests.calibration.measure import (
    BASELINE_PATH,
    RESULTS_SCHEMA,
    load_baseline,
    write_json,
)
from tests.calibration.report import render_summary
from tests.calibration.scoring import Trial, combine_scenarios, rates


def describe_environment() -> dict[str, str]:
    """What the numbers depend on besides the code: the interpreter and the maths."""
    return {
        "python": platform.python_version(),
        **{
            name: version(name)
            for name in ("numpy", "scipy", "statsmodels", "scikit-learn", "polars")
        },
    }


@pytest.fixture(scope="session")
def measured(pytestconfig: pytest.Config) -> Iterator[dict[str, list[Trial]]]:
    """The trials of every scenario that ran, by scenario name.

    The results are written when the session ends, even if a test failed, so a
    regression can be inspected. ``--update-calibration-baseline`` merges the
    scenarios that ran into the baseline and keeps the rest.
    """
    scenarios: dict[str, list[Trial]] = {}
    yield scenarios
    if not scenarios:
        return

    by_scenario = {name: rates(trials) for name, trials in scenarios.items()}
    results: dict[str, Any] = {
        "schema": RESULTS_SCHEMA,
        "environment": describe_environment(),
        "checks": combine_scenarios(by_scenario),
        "scenarios": by_scenario,
    }
    baseline = load_baseline()
    # The invocation directory, not the temporary CWD the suite runs from.
    path = Path(pytestconfig.invocation_params.dir) / pytestconfig.getoption(
        "--calibration-results"
    )
    write_json(path, results)
    path.with_suffix(".md").write_text(render_summary(results, baseline))

    if pytestconfig.getoption("--update-calibration-baseline"):
        every = {**baseline["scenarios"], **by_scenario}
        write_json(
            BASELINE_PATH,
            {
                "schema": RESULTS_SCHEMA,
                "environment": results["environment"],
                "checks": combine_scenarios(every),
                "scenarios": every,
            },
        )
