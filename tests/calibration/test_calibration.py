"""Measure every check on planted-truth frames and compare with the baseline.

Run with ``uv run pytest -m calibration``; see ``tests/calibration/README.md``. Each
scenario is profiled over its seeds, and a check whose detection or false-alarm rate
got worse than ``baseline.json`` (beyond the noise of the seeds) fails the scenario.
"""

from __future__ import annotations

import pytest

from tests.calibration.measure import load_baseline, run_trial
from tests.calibration.scenario import Scenario
from tests.calibration.scenarios import SCENARIOS
from tests.calibration.scoring import Trial, rates, regressions

pytestmark = pytest.mark.calibration


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=list(SCENARIOS))
def test_scenario(
    scenario: Scenario,
    measured: dict[str, list[Trial]],
    pytestconfig: pytest.Config,
) -> None:
    trials = [t for seed in range(scenario.seeds) for t in run_trial(scenario, seed)]
    measured[scenario.name] = trials
    found = rates(trials)

    never_ran = [
        check_id
        for check_id, entry in found.items()
        if entry["plants"] == 0 and entry["trials"] == 0 and entry["unscored"]
    ]
    planted = {t.check_id for t in trials if t.planted}
    silent = sorted(c for c in planted if found[c]["trials"] == 0)
    assert not silent, (
        f"{scenario.name} plants flaws for checks that did not run on its frame "
        f"({', '.join(silent)}); their statuses: "
        + ", ".join(f"{c} {found[c]['unscored']}" for c in silent)
    )
    del never_ran

    if pytestconfig.getoption("--update-calibration-baseline"):
        return
    baseline = load_baseline()["scenarios"].get(scenario.name)
    assert baseline is not None, (
        f"{scenario.name} has no baseline; record it with "
        "`pytest -m calibration --update-calibration-baseline`"
    )
    problems = regressions(found, baseline)
    assert not problems, "\n".join(problems)
