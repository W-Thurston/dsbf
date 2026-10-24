"""The scoring of findings against plants, on results small enough to read."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from dsbf.checks import core_pack
from dsbf.model import CheckStatus, Severity
from tests.calibration.scenarios import SCENARIOS
from tests.calibration.scoring import (
    Planted,
    Probe,
    combine_scenarios,
    plant,
    rates,
    regressions,
    score,
    wilson,
)

pytestmark = pytest.mark.unit


def finding(code: str, severity: Severity, *columns: str) -> Any:
    return SimpleNamespace(
        code=code, severity=severity, subject=SimpleNamespace(columns=columns)
    )


def result(
    findings: list[Any] | None = None,
    status: CheckStatus = CheckStatus.OK,
    version: str = "1.0.0",
    output: dict[str, Any] | None = None,
) -> Any:
    return SimpleNamespace(
        check_version=version,
        status=status,
        findings=tuple(findings or ()),
        output=output,
    )


def run_of(**results: Any) -> Any:
    return SimpleNamespace(results=results)


def trial_of(planted: Planted, check_id: str, **results: Any) -> Any:
    trials = score("s", 0, planted, run_of(**results), [check_id])
    assert len(trials) == 1
    return trials[0]


def test_a_warning_on_the_planted_columns_is_a_detection() -> None:
    planted = Planted(None, plants=(plant("outliers", "x"),))
    trial = trial_of(
        planted, "outliers", outliers=result([finding("o", Severity.WARN, "x")])
    )
    assert (trial.planted, trial.detected, trial.false_alarms) == (1, 1, 0)


def test_the_columns_of_a_pair_match_in_either_order() -> None:
    planted = Planted(None, plants=(plant("associations", "a", "b"),))
    trial = trial_of(
        planted,
        "associations",
        associations=result([finding("s", Severity.WARN, "b", "a")]),
    )
    assert trial.detected == 1


def test_a_finding_below_a_warning_detects_a_plant_without_being_an_alarm() -> None:
    planted = Planted(None, plants=(plant("string_anomalies", "x"),))
    trial = trial_of(
        planted,
        "string_anomalies",
        string_anomalies=result([finding("e", Severity.INFO, "x")]),
    )
    assert (trial.detected, trial.alarms, trial.false_alarms) == (1, 0, 0)


def test_a_plant_can_ask_for_a_severity() -> None:
    planted = Planted(None, plants=(plant("c", "x", min_severity=Severity.WARN),))
    trial = trial_of(planted, "c", c=result([finding("e", Severity.NOTICE, "x")]))
    assert trial.detected == 0


def test_an_unexplained_warning_is_a_false_alarm_and_an_allowed_one_is_not() -> None:
    planted = Planted(None, plants=(plant("c", "x"),), allowed=(plant("c", "y"),))
    found = [
        finding("a", Severity.WARN, "x"),
        finding("a", Severity.WARN, "y"),
        finding("a", Severity.WARN, "z"),
        finding("a", Severity.INFO, "w"),
    ]
    trial = trial_of(planted, "c", c=result(found))
    assert (trial.detected, trial.alarms, trial.false_alarms) == (1, 3, 1)


def test_the_codes_of_a_plant_limit_what_it_explains() -> None:
    planted = Planted(None, plants=(plant("c", "x", codes=["right"]),))
    trial = trial_of(planted, "c", c=result([finding("wrong", Severity.WARN, "x")]))
    assert (trial.detected, trial.false_alarms) == (0, 1)


def test_a_plant_without_columns_matches_any_subject() -> None:
    planted = Planted(None, plants=(plant("duplicate_rows"),))
    trial = trial_of(
        planted,
        "duplicate_rows",
        duplicate_rows=result([finding("d", Severity.WARN, "a", "b")]),
    )
    assert trial.detected == 1


def test_an_info_only_check_claims_with_every_finding() -> None:
    trial = trial_of(
        Planted(None),
        "normality",
        normality=result([finding("non_normal", Severity.INFO, "x")]),
    )
    assert trial.false_alarms == 1


def test_a_check_that_names_its_claim_codes_ignores_the_others() -> None:
    found = [
        finding("missingness_unexplained", Severity.INFO, "x"),
        finding("missingness_depends_on_observed", Severity.INFO, "y"),
    ]
    trial = trial_of(
        Planted(None), "missingness_mechanism", missingness_mechanism=result(found)
    )
    assert trial.false_alarms == 1


def test_advice_is_scored_only_where_the_scenario_says_which_is_right() -> None:
    run = run_of(imputation=result([finding("i", Severity.INFO, "x")]))
    assert score("s", 0, Planted(None), run, ["imputation"]) == []
    planted = Planted(None, plants=(plant("imputation", "x"),))
    assert score("s", 0, planted, run, ["imputation"])[0].detected == 1


def test_a_check_that_did_not_end_ok_is_not_scored() -> None:
    planted = Planted(None, plants=(plant("c", "x"),))
    trial = trial_of(
        planted, "c", c=result(status=CheckStatus.NOT_APPLICABLE, findings=[])
    )
    found = rates([trial])["c"]
    assert found["trials"] == 0
    assert found["unscored"] == {"not_applicable": 1}
    assert "detection_rate" not in found


def test_a_probe_reads_a_number_from_the_result() -> None:
    def size(result: Any, planted: Planted) -> float:
        return result.output["n"] / 10

    planted = Planted(None, probes=(Probe("c", "size", size),))
    trial = trial_of(planted, "c", c=result(output={"n": 4}))
    assert trial.metrics == {"size": 0.4}
    assert rates([trial])["c"]["metrics"]["size"] == {
        "mean": 0.4,
        "n": 1,
        "higher_is_better": True,
    }


def test_rates_count_plants_and_false_alarm_trials() -> None:
    planted = Planted(None, plants=(plant("c", "x"),))
    hit = trial_of(planted, "c", c=result([finding("a", Severity.WARN, "x")]))
    miss = trial_of(planted, "c", c=result([finding("a", Severity.WARN, "y")]))
    found = rates([hit, miss])["c"]
    assert found["detection_rate"] == 0.5
    assert found["false_alarm_rate"] == 0.5
    assert found["trials"] == 2


def test_wilson_interval_of_no_successes() -> None:
    low, high = wilson(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)


def _entry(**given: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": "1.0.0",
        "trials": 20,
        "plants": 20,
        "detected": 20,
        "false_alarm_trials": 0,
        "unscored": {},
        "detection_rate": 1.0,
        "detection_interval": [0.84, 1.0],
        "false_alarm_rate": 0.0,
        "false_alarm_interval": [0.0, 0.16],
    }
    return {**base, **given}


def test_a_detection_rate_that_fell_beyond_the_noise_is_a_regression() -> None:
    baseline = {"c": _entry()}
    assert regressions({"c": _entry(detection_rate=0.95)}, baseline) == []
    problems = regressions({"c": _entry(detection_rate=0.7)}, baseline)
    assert len(problems) == 1 and "detection fell" in problems[0]


def test_a_false_alarm_rate_that_rose_beyond_the_noise_is_a_regression() -> None:
    baseline = {"c": _entry()}
    assert regressions({"c": _entry(false_alarm_rate=0.1)}, baseline) == []
    problems = regressions({"c": _entry(false_alarm_rate=0.4)}, baseline)
    assert len(problems) == 1 and "false alarms rose" in problems[0]


def test_better_rates_are_not_regressions() -> None:
    baseline = {"c": _entry(detection_rate=0.5, false_alarm_rate=0.5)}
    assert regressions({"c": _entry()}, baseline) == []


def test_a_new_version_asks_for_a_fresh_baseline() -> None:
    problems = regressions({"c": _entry(version="1.1.0")}, {"c": _entry()})
    assert problems and "refresh the baseline" in problems[0]


def test_a_probe_that_moved_the_wrong_way_is_a_regression() -> None:
    def with_probe(mean: float, better: bool) -> dict[str, Any]:
        return _entry(
            metrics={"p": {"mean": mean, "n": 10, "higher_is_better": better}}
        )

    assert regressions({"c": with_probe(0.8, True)}, {"c": with_probe(0.9, True)})
    assert not regressions({"c": with_probe(0.88, True)}, {"c": with_probe(0.9, True)})
    assert regressions({"c": with_probe(0.3, False)}, {"c": with_probe(0.2, False)})


def test_scenarios_combine_into_the_rates_of_all_of_them() -> None:
    first = {"c": _entry(trials=10, plants=10, detected=10)}
    second = {"c": _entry(trials=30, plants=30, detected=15)}
    whole = combine_scenarios({"a": first, "b": second})["c"]
    assert (whole["trials"], whole["plants"], whole["detected"]) == (40, 40, 25)
    assert whole["detection_rate"] == pytest.approx(0.625)


def test_every_scenario_is_reproducible_and_names_real_checks() -> None:
    known = {c.spec.id for c in core_pack().checks}
    for name, scenario in SCENARIOS.items():
        first, second = scenario.build(1), scenario.build(1)
        pd.testing.assert_frame_equal(first.frame, second.frame, obj=name)
        assert [(p.check_id, p.columns, p.codes) for p in first.plants] == [
            (p.check_id, p.columns, p.codes) for p in second.plants
        ]
        named = {p.check_id for p in (*first.plants, *first.allowed)}
        named |= {p.check_id for p in first.probes} | set(first.checks or ())
        assert named <= known, f"{name} names unknown checks {sorted(named - known)}"
        assert scenario.summary, f"{name} has no docstring"
        assert all(p.check_id in (first.checks or known) for p in first.plants), (
            f"{name} plants a flaw for a check it does not run"
        )
