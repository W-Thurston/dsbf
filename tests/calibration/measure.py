"""Profile every scenario over its seeds and score the checks.

The measurement is `dsbf.profile()` itself, in this process, on frames that
`tests.calibration.scenarios` generates from a seed alone. Nothing here depends on
the pytest session, so a rate can be probed from a script:

    uv run python -m tests.calibration.measure --scenario noise_numeric --seeds 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from dsbf.checks import core_pack
from dsbf.data import load
from dsbf.engine import Reference, Registry, Selection, execute, make_plan
from dsbf.model import Run
from dsbf.profiling import profile_dataset
from tests.calibration.scenario import Scenario
from tests.calibration.scenarios import SCENARIOS
from tests.calibration.scoring import (
    Alarm,
    Planted,
    Trial,
    combine_scenarios,
    rates,
    score,
)

CALIBRATION_DIR = Path(__file__).parent
BASELINE_PATH = CALIBRATION_DIR / "baseline.json"
RESULTS_SCHEMA = 1

ALL_CHECKS = tuple(sorted(c.spec.id for c in core_pack().checks))


def run_trial(
    scenario: Scenario, seed: int, only: Sequence[str] | None = None
) -> list[Trial]:
    """Generate the frame of ``scenario`` for ``seed``, profile it and score it.

    The core pack runs at the deep depth through the engine, as `dsbf.profile()`
    does, but with the check parameters of the scenario (the library takes none).
    """
    planted = scenario.build(seed)
    return score(
        scenario.name, seed, planted, _run(scenario, planted), _wanted(planted)
    )


def _wanted(planted: Planted) -> tuple[str, ...]:
    return planted.checks or ALL_CHECKS


def _closure(wanted: Iterable[str]) -> set[str]:
    """``wanted`` and the checks they require, which the plan cannot do without."""
    specs = {c.spec.id: c.spec for c in core_pack().checks}
    pending, found = list(wanted), set()
    while pending:
        check_id = pending.pop()
        if check_id not in found:
            found.add(check_id)
            pending.extend(specs[check_id].requires.checks)
    return found


def _run(scenario: Scenario, planted: Planted) -> Run:
    """Profile the frame of ``planted`` and run the checks it asks for."""
    wanted = _wanted(planted)
    dataset = load(planted.frame, name=scenario.name)
    data_profile = profile_dataset(dataset)
    reference = None
    if planted.reference is not None:
        ref_data = load(planted.reference, name=f"{scenario.name}-reference")
        reference = Reference.of_data(ref_data, profile_dataset(ref_data))
    registry = Registry(core_pack().checks)
    given = {
        k: v
        for k, v in (("target", planted.target), ("time_index", planted.time_index))
        if v is not None
    }
    params: dict[str, dict[str, Any]] = {}
    for check in registry:
        fields = check.spec.params.model_fields
        merged = {
            **{k: v for k, v in given.items() if k in fields},
            **planted.params.get(check.spec.id, {}),
        }
        if merged:
            params[check.spec.id] = merged
    plan = make_plan(
        registry,
        Selection(
            packs=["core"],
            depth="deep",
            exclude=[c for c in ALL_CHECKS if c not in _closure(wanted)],
        ),
        params=params,
        profile=data_profile,
        has_reference=reference is not None,
    )
    return execute(
        plan, dataset, data_profile, dsbf_version="calibration", reference=reference
    )


def explain(name: str, seed: int = 0) -> list[str]:
    """Every finding of the scenario at ``seed``, marked as a hit, allowed or false."""
    scenario = SCENARIOS[name]
    planted = scenario.build(seed)
    run = _run(scenario, planted)
    lines = [f"frame {planted.frame.shape}"]
    for check_id, result in sorted(run.results.items()):
        if result.status.value != "ok":
            lines.append(f"{check_id}: {result.status.value}")
        for finding in result.findings:
            alarm = Alarm(check_id, frozenset(finding.subject.columns), finding)
            mark = "  "
            if any(p.explains(alarm) for p in planted.plants):
                mark = "OK"
            elif any(p.explains(alarm) for p in planted.allowed):
                mark = "~~"
            elif alarm.claim:
                mark = "!!"
            lines.append(
                f"{mark} {check_id} {finding.code} {finding.severity.value} "
                f"{list(finding.subject.columns)} {finding.title}"
            )
    for p in planted.plants:
        lines.append(f"plant {p.check_id} {sorted(p.columns or [])} {sorted(p.codes)}")
    return lines


def _scenario_trials(
    name: str, seeds: Sequence[int], only: Sequence[str] | None = None
) -> list[Trial]:
    scenario = SCENARIOS[name]
    return [t for seed in seeds for t in run_trial(scenario, seed, only)]


def measure(
    names: Iterable[str] | None = None,
    seeds: int | None = None,
    workers: int = 1,
    only: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The rates of every check, overall and by scenario.

    ``seeds`` replaces each scenario's own count (to probe quickly); ``workers`` runs
    scenarios in that many processes; ``only`` runs just those checks (and what they
    require), for a quick look at one.
    """
    chosen = list(names) if names else list(SCENARIOS)
    started = time.perf_counter()
    jobs = {n: range(seeds or SCENARIOS[n].seeds) for n in chosen}
    if workers > 1:
        # Fresh interpreters: a check that runs in a process of its own cannot start
        # one from a forked worker that holds the engine's threads.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as pool:
            futures = {
                n: pool.submit(_scenario_trials, n, list(s), only)
                for n, s in jobs.items()
            }
            per_scenario = {n: f.result() for n, f in futures.items()}
    else:
        per_scenario = {n: _scenario_trials(n, list(s), only) for n, s in jobs.items()}
    by_scenario = {n: rates(items) for n, items in per_scenario.items()}
    return {
        "schema": RESULTS_SCHEMA,
        "seconds": round(time.perf_counter() - started, 1),
        "checks": combine_scenarios(by_scenario),
        "scenarios": by_scenario,
    }


def load_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.exists():
        return {"schema": RESULTS_SCHEMA, "checks": {}, "scenarios": {}}
    return json.loads(BASELINE_PATH.read_text())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True)
    # An interval is two numbers; one line each is noise in a diff.
    text = re.sub(r"\[\s+([-\d.e]+),\s+([-\d.e]+)\s+\]", r"[\1, \2]", text)
    path.write_text(text + "\n")


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", help="a scenario (repeatable)")
    parser.add_argument("--seeds", type=int, help="seeds per scenario")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--check", action="append", help="show only these checks")
    parser.add_argument(
        "--only", action="append", help="run only this check (repeatable)"
    )
    parser.add_argument("--output", type=Path, help="write the results JSON here")
    parser.add_argument(
        "--explain", type=int, help="print the findings of one scenario at this seed"
    )
    args = parser.parse_args(argv)
    if args.explain is not None:
        for name in args.scenario or []:
            print("\n".join(explain(name, args.explain)))
        return 0
    result = measure(args.scenario, args.seeds, args.workers, args.only)
    if args.output:
        write_json(args.output, result)
    shown = result["checks"]
    if args.check:
        shown = {k: v for k, v in shown.items() if k in args.check}
    print(f"{result['seconds']} s")
    for check_id, entry in shown.items():
        print(check_id, json.dumps(entry))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
