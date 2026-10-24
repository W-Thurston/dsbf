"""Run DSBF on one scenario in an isolated child process and collect what it cost."""

from __future__ import annotations

import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from tests.perf.scenarios import EngineScenario, ProfileScenario, Scenario

PERF_DIR = Path(__file__).parent
BASELINE_PATH = PERF_DIR / "baseline.json"
RESULTS_SCHEMA = 1
# The slowest scenario took about twelve minutes before any optimisation.
RUN_TIMEOUT_SECONDS = 3600
# Some tasks walk sets of strings, and that order follows PYTHONHASHSEED; pin it so the
# amount of work, not just its speed, is the same in every run.
HASH_SEED = "0"


def _run_child(command: list[str], workdir: Path, name: str) -> dict[str, Any]:
    """Run a measuring script in a fresh interpreter and return the JSON it wrote."""
    env = {
        **os.environ,
        "PYTHONHASHSEED": HASH_SEED,
        "DSBF_DB_PATH": str(workdir / "dsbf.db"),
        "DSBF_HOME": str(workdir / "home"),
    }
    completed = subprocess.run(
        command,
        cwd=workdir / "cwd",
        env=env,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Measuring {name} exited with {completed.returncode}:\n"
            f"{completed.stderr[-4000:]}"
        )
    return json.loads((workdir / "out" / "perf_run.json").read_text())


def run_once(
    scenario: Scenario, workdir: Path, cprofile: bool = False
) -> dict[str, Any]:
    """Profile ``scenario`` once under ``workdir`` and return its measurements."""
    workdir = workdir.resolve()
    data_dir, out_dir, cwd = workdir / "data", workdir / "out", workdir / "cwd"
    for directory in (data_dir, out_dir, cwd):
        directory.mkdir(parents=True, exist_ok=True)

    csv_path = scenario.write_csv(data_dir).resolve()
    command = [
        sys.executable,
        str(PERF_DIR / "measure_one.py"),
        scenario.name,
        str(csv_path),
        str(out_dir),
    ]
    if scenario.time_series_column:
        command += ["--time-series", scenario.time_series_column]
    if cprofile:
        command.append("--cprofile")

    return _run_child(command, workdir, name=scenario.name)


def run_profile_once(
    scenario: ProfileScenario, data_path: Path, workdir: Path
) -> dict[str, Any]:
    """Run the profile pass over ``data_path`` once under ``workdir``."""
    workdir = workdir.resolve()
    for directory in (workdir / "out", workdir / "cwd"):
        directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PERF_DIR / "measure_profile.py"),
        scenario.name,
        str(data_path.resolve()),
        str(workdir / "out"),
    ]
    return _run_child(command, workdir, name=scenario.name)


def run_engine_once(scenario: EngineScenario, workdir: Path) -> dict[str, Any]:
    """Call ``dsbf.profile()`` on ``scenario``'s data once under ``workdir``."""
    workdir = workdir.resolve()
    data_dir, out_dir, cwd = workdir / "data", workdir / "out", workdir / "cwd"
    for directory in (data_dir, out_dir, cwd):
        directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PERF_DIR / "measure_engine.py"),
        scenario.name,
        str(scenario.write_csv(data_dir).resolve()),
        str(out_dir),
        "--workers",
        "default" if scenario.workers is None else str(scenario.workers),
        "--depth",
        scenario.depth,
    ]
    if scenario.time_series_column:
        command += ["--time-index", scenario.time_series_column]
    return _run_child(command, workdir, name=scenario.name)


def measure(
    scenario: Scenario | ProfileScenario | EngineScenario,
    workdir: Path,
    cprofile: bool = False,
) -> dict[str, Any]:
    """Run ``scenario`` ``repeats`` times and keep the run with the median total."""
    if isinstance(scenario, ProfileScenario):
        # The data is written once; each repeat is a fresh interpreter over it.
        data_dir = workdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = scenario.write_data(data_dir)
        runs = [
            run_profile_once(scenario, data_path, workdir / f"run{i}")
            for i in range(scenario.repeats)
        ]
    elif isinstance(scenario, EngineScenario):
        runs = [
            run_engine_once(scenario, workdir / f"run{i}")
            for i in range(scenario.repeats)
        ]
    else:
        runs = [
            run_once(scenario, workdir / f"run{i}", cprofile)
            for i in range(scenario.repeats)
        ]
    median_total = statistics.median_low(r["total_seconds"] for r in runs)
    chosen = next(r for r in runs if r["total_seconds"] == median_total)
    chosen["repeats"] = len(runs)
    chosen["total_seconds_all_runs"] = [r["total_seconds"] for r in runs]
    return chosen


def describe_environment() -> dict[str, Any]:
    """Where and on what the numbers were taken."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PERF_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ci": os.environ.get("GITHUB_ACTIONS") == "true",
        "commit": commit,
        "recorded_at": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }


def load_results(path: Path) -> dict[str, Any]:
    """Load a results or baseline file, or an empty one if it does not exist."""
    if not path.exists():
        return {"schema": RESULTS_SCHEMA, "environment": {}, "scenarios": {}}
    return json.loads(path.read_text())


def write_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
