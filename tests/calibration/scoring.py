"""Score the findings of a run against what was planted in the data.

A scenario plants a known flaw (or none), a run profiles the frame, and the findings
are compared with the plants. A check is *scored* on a frame only when it ended `ok`: a
check that was not applicable or failed is counted apart, never as a miss.

- An *alarm* is a finding of severity `warn` or `error`; the notices and info findings
  that advise rather than accuse are not claims about the data, except for the checks
  in `INFO_CLAIMS`, which report nothing above `info`. The checks in
  `ADVICE_CHECKS` only advise, so for them any finding is the claim being scored (a
  plant says which advice is right, with `where`), and they are scored only on frames
  whose scenario declares the advice that is right.
- A plant is *detected* when a finding of its check, at the severity the plant asks
  for, is about the planted columns (compared as sets, so the two columns of a pair
  match in either order). Any finding counts, not only an alarm: a check that says
  something at `info` about a planted flaw has found it.
- A *false alarm* is an alarm that no plant and no allowance of its check explains.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from dsbf.model import CheckResult, CheckStatus, Finding, Run, Severity

ALARM = Severity.WARN
# Where a check's claims start when that is below `warn`.
CLAIM_FLOORS: dict[str, Severity] = {"reference_drift": Severity.NOTICE}
ADVICE_CHECKS = frozenset(
    {
        "categorical_encoding",
        "datetime_features",
        "dtype_optimizations",
        "imputation",
        "numeric_binning",
        "transformations",
    }
)
# Checks whose finding is the claim whatever its severity, and the codes that are
# (`None`: every code). They report at `info` only, or say it at `info` until it is
# strong.
INFO_CLAIMS: dict[str, frozenset[str] | None] = {
    "autocorrelation": None,
    "encoded_columns": None,
    "group_differences": None,
    "missingness_mechanism": frozenset({"missingness_depends_on_observed"}),
    "multivariate_outliers": None,
    "mutual_information": None,
    "normality": None,
    "seasonal_decomposition": None,
}
Z95 = 1.959964


@dataclass(frozen=True)
class Plant:
    """A flaw that ``check_id`` must raise an alarm about.

    ``columns`` are the columns the alarm's subject names (`None`: any subject, for a
    flaw of the dataset as a whole). ``codes`` limits which finding codes count; empty
    means any code. ``where`` limits it to findings the function accepts.
    ``min_severity`` is the least severity at which the check has said enough.
    """

    check_id: str
    columns: frozenset[str] | None = None
    codes: frozenset[str] = frozenset()
    where: Callable[[Finding], bool] | None = None
    min_severity: Severity = Severity.INFO

    def explains(self, alarm: Alarm) -> bool:
        if alarm.check_id != self.check_id:
            return False
        if alarm.finding.severity < self.min_severity:
            return False
        if self.codes and alarm.finding.code not in self.codes:
            return False
        if self.columns is not None and self.columns != alarm.columns:
            return False
        return self.where is None or self.where(alarm.finding)


def plant(
    check_id: str,
    *columns: str,
    codes: Iterable[str] = (),
    where: Callable[[Finding], bool] | None = None,
    min_severity: Severity = Severity.INFO,
) -> Plant:
    """A plant on ``columns``; with no column, on the dataset as a whole."""
    return Plant(
        check_id,
        frozenset(columns) if columns else None,
        frozenset(codes),
        where,
        min_severity,
    )


@dataclass(frozen=True)
class Alarm:
    """A finding of a check, and whether it is a claim that the data is flawed."""

    check_id: str
    columns: frozenset[str]
    finding: Finding

    @property
    def claim(self) -> bool:
        return is_claim(self.check_id, self.finding)


@dataclass(frozen=True)
class Probe:
    """A number read off a check's result, for what a detection rate cannot say.

    ``measure(result, planted)`` returns the number, or `None` when the check says
    nothing about it on this frame. The mean over trials is reported and compared
    with the baseline; ``higher_is_better`` says which way a change is a regression.
    """

    check_id: str
    name: str
    measure: Callable[[CheckResult, Planted], float | None]
    higher_is_better: bool = True


@dataclass
class Planted:
    """One generated frame with its truth and the settings its run needs."""

    frame: Any
    plants: tuple[Plant, ...] = ()
    allowed: tuple[Plant, ...] = ()
    """Alarms that are consequences of the plants, so are neither hits nor misses."""
    checks: tuple[str, ...] | None = None
    """The checks to run; `None` runs the whole default selection."""
    target: str | None = None
    time_index: str | None = None
    reference: Any = None
    """The frame the checks that compare with a reference are given."""
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Parameters for individual checks, by check id."""
    probes: tuple[Probe, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    """Truth that the probes need and a plant cannot say, such as planted row pairs."""


@dataclass(frozen=True)
class Trial:
    """What one check did on one generated frame."""

    scenario: str
    seed: int
    check_id: str
    version: str
    status: str
    planted: int
    detected: int
    alarms: int
    false_alarms: int
    metrics: dict[str, float] = field(default_factory=dict)
    directions: dict[str, bool] = field(default_factory=dict)


def is_claim(check_id: str, finding: Finding) -> bool:
    """Whether ``finding`` is an alarm, or (for advice) a piece of advice to judge."""
    if check_id in ADVICE_CHECKS:
        return True
    if check_id in INFO_CLAIMS:
        codes = INFO_CLAIMS[check_id]
        if codes is None or finding.code in codes:
            return True
    return finding.severity >= CLAIM_FLOORS.get(check_id, ALARM)


def alarms_of(run: Run) -> list[Alarm]:
    """Every finding of every check in ``run``, claim or not."""
    return [
        Alarm(check_id, frozenset(finding.subject.columns), finding)
        for check_id, result in run.results.items()
        for finding in result.findings
    ]


def score(
    scenario: str, seed: int, planted: Planted, run: Run, checks: Iterable[str]
) -> list[Trial]:
    """One `Trial` for each of ``checks`` that ``run`` holds a result for."""
    alarms = alarms_of(run)
    trials = []
    for check_id in sorted(set(checks)):
        result = run.results.get(check_id)
        if result is None:
            continue
        plants = [p for p in planted.plants if p.check_id == check_id]
        allowed = [p for p in planted.allowed if p.check_id == check_id]
        if check_id in ADVICE_CHECKS and not (plants or allowed):
            # Advice is given for every column; only a scenario that says which
            # advice is right can tell a wrong one.
            continue
        own = [a for a in alarms if a.check_id == check_id]
        detected = sum(1 for p in plants if any(p.explains(a) for a in own))
        false_alarms = sum(
            1
            for a in own
            if a.claim and not any(p.explains(a) for p in (*plants, *allowed))
        )
        metrics: dict[str, float] = {}
        directions: dict[str, bool] = {}
        for probe in planted.probes:
            if probe.check_id != check_id:
                continue
            value = probe.measure(result, planted)
            if value is not None:
                metrics[probe.name] = float(value)
                directions[probe.name] = probe.higher_is_better
        trials.append(
            Trial(
                scenario=scenario,
                seed=seed,
                check_id=check_id,
                version=result.check_version,
                status=result.status.value,
                planted=len(plants),
                detected=detected,
                alarms=sum(1 for a in own if a.claim),
                false_alarms=false_alarms,
                metrics=metrics,
                directions=directions,
            )
        )
    return trials


def wilson(successes: int, trials: int) -> tuple[float, float]:
    """The 95% Wilson score interval for a rate; (0, 1) when there are no trials."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    z2 = Z95**2
    centre = (p + z2 / (2 * trials)) / (1 + z2 / trials)
    half = (
        Z95 * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials**2)) / (1 + z2 / trials)
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def rates(trials: Iterable[Trial]) -> dict[str, dict[str, Any]]:
    """Per check: its version, how often it found the plants and how often it misfired.

    `detection_rate` is the share of plants detected (absent when none were planted
    where the check ran); `false_alarm_rate` the share of scored trials with at least
    one false alarm. Each carries its 95% interval. `unscored` counts the trials where
    the check did not end `ok`, with their statuses, and `metrics` the mean of each
    probe.
    """
    grouped: dict[str, list[Trial]] = defaultdict(list)
    for trial in trials:
        grouped[trial.check_id].append(trial)
    report: dict[str, dict[str, Any]] = {}
    for check_id, items in sorted(grouped.items()):
        scored = [t for t in items if t.status == CheckStatus.OK.value]
        statuses: dict[str, int] = defaultdict(int)
        for t in items:
            if t.status != CheckStatus.OK.value:
                statuses[t.status] += 1
        series: dict[str, list[float]] = defaultdict(list)
        better: dict[str, bool] = {}
        for t in scored:
            for name, value in t.metrics.items():
                series[name].append(value)
                better[name] = t.directions[name]
        report[check_id] = finish(
            {
                "version": items[-1].version,
                "trials": len(scored),
                "plants": sum(t.planted for t in scored),
                "detected": sum(t.detected for t in scored),
                "false_alarm_trials": sum(1 for t in scored if t.false_alarms),
                "unscored": dict(sorted(statuses.items())),
                "metrics": {
                    name: {
                        "mean": sum(values) / len(values),
                        "n": len(values),
                        "higher_is_better": better[name],
                    }
                    for name, values in sorted(series.items())
                },
            }
        )
    return report


def finish(counts: dict[str, Any]) -> dict[str, Any]:
    """``counts`` with the rates and intervals worked out from them."""
    entry = {k: v for k, v in counts.items() if k != "metrics"}
    if counts["plants"]:
        entry["detection_rate"] = round(counts["detected"] / counts["plants"], 4)
        entry["detection_interval"] = [
            round(v, 4) for v in wilson(counts["detected"], counts["plants"])
        ]
    if counts["trials"]:
        entry["false_alarm_rate"] = round(
            counts["false_alarm_trials"] / counts["trials"], 4
        )
        entry["false_alarm_interval"] = [
            round(v, 4) for v in wilson(counts["false_alarm_trials"], counts["trials"])
        ]
    if counts["metrics"]:
        entry["metrics"] = {
            name: {**metric, "mean": round(metric["mean"], 4)}
            for name, metric in counts["metrics"].items()
        }
    return entry


def combine(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """One check's entries from several scenarios, as the entry of all of them."""
    entries = list(entries)
    counts: dict[str, Any] = {
        "version": entries[-1]["version"],
        "trials": sum(e["trials"] for e in entries),
        "plants": sum(e["plants"] for e in entries),
        "detected": sum(e["detected"] for e in entries),
        "false_alarm_trials": sum(e["false_alarm_trials"] for e in entries),
        "unscored": {},
        "metrics": {},
    }
    for e in entries:
        for status, count in e["unscored"].items():
            counts["unscored"][status] = counts["unscored"].get(status, 0) + count
        for name, metric in e.get("metrics", {}).items():
            total = counts["metrics"].setdefault(
                name,
                {"mean": 0.0, "n": 0, "higher_is_better": metric["higher_is_better"]},
            )
            total["mean"] = (
                total["mean"] * total["n"] + metric["mean"] * metric["n"]
            ) / (total["n"] + metric["n"])
            total["n"] += metric["n"]
    counts["unscored"] = dict(sorted(counts["unscored"].items()))
    return finish(counts)


def combine_scenarios(
    scenarios: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """The entry of every check over all of ``scenarios`` (name -> check -> entry)."""
    by_check: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for per_check in scenarios.values():
        for check_id, entry in per_check.items():
            by_check[check_id].append(entry)
    return {check_id: combine(items) for check_id, items in sorted(by_check.items())}


METRIC_MARGIN = 0.05
"""How far a probe's mean may move the wrong way before it counts as a regression."""
MARGIN = 0.02
"""Slack beyond the baseline's own interval before a rate counts as a regression."""
MAX_HALF_WIDTH = 0.15
"""Most of an interval that counts as slack: a rate over three seeds is exact."""


def _slack(interval: Iterable[float]) -> float:
    low, high = interval
    return min((high - low) / 2, MAX_HALF_WIDTH) + MARGIN


def regressions(
    measured: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """What got worse than ``baseline``, one line each; empty when nothing did.

    A rate that moved by less than the baseline's interval plus `MARGIN` is noise. A
    check whose version differs is reported as stale, whatever its rates.
    """
    problems: list[str] = []
    for check_id, was in sorted(baseline.items()):
        now = measured.get(check_id)
        if now is None:
            problems.append(f"{check_id}: in the baseline but not measured")
            continue
        if now["version"] != was["version"]:
            problems.append(
                f"{check_id}: version {was['version']} in the baseline, "
                f"{now['version']} measured; refresh the baseline"
            )
            continue
        if "detection_rate" in was:
            slack = _slack(was["detection_interval"])
            if now.get("detection_rate", 0.0) < was["detection_rate"] - slack:
                problems.append(
                    f"{check_id}: detection fell from {was['detection_rate']:.2f} "
                    f"to {now.get('detection_rate', 0.0):.2f}"
                )
        if "false_alarm_rate" in was:
            slack = _slack(was["false_alarm_interval"])
            if now.get("false_alarm_rate", 0.0) > was["false_alarm_rate"] + slack:
                problems.append(
                    f"{check_id}: false alarms rose from "
                    f"{was['false_alarm_rate']:.2f} to "
                    f"{now.get('false_alarm_rate', 0.0):.2f}"
                )
        for name, was_metric in was.get("metrics", {}).items():
            now_metric = now.get("metrics", {}).get(name)
            if now_metric is None:
                problems.append(f"{check_id}: {name} is in the baseline, not measured")
                continue
            drop = (now_metric["mean"] - was_metric["mean"]) * (
                1 if was_metric["higher_is_better"] else -1
            )
            if drop < -METRIC_MARGIN:
                problems.append(
                    f"{check_id}: {name} moved from {was_metric['mean']:.3f} "
                    f"to {now_metric['mean']:.3f}"
                )
    return problems
