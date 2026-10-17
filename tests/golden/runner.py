"""Run DSBF on one dataset in an isolated child process and load what it wrote."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.golden.datasets import GOLDEN_DIR, GoldenDataset

# Runs that must reproduce byte for byte need a fixed string hash seed: some tasks walk
# sets of strings, and that order follows PYTHONHASHSEED.
HASH_SEED = "0"
RUN_TIMEOUT_SECONDS = 1800


@dataclass(frozen=True)
class GoldenRun:
    report: dict[str, Any]
    run_info: dict[str, Any]
    roots: list[str]


def profile(dataset: GoldenDataset, workdir: Path) -> GoldenRun:
    """Profile ``dataset`` under ``workdir`` and return its own report.

    The report is read from this run's output directory, never found by searching, so it
    cannot be a stale run of another dataset.
    """
    workdir = workdir.resolve()
    data_dir = workdir / "data"
    out_dir = workdir / "out"
    cwd = workdir / "cwd"
    for directory in (data_dir, out_dir, cwd):
        directory.mkdir(parents=True, exist_ok=True)

    csv_path = dataset.write_csv(data_dir).resolve()
    env = {
        **os.environ,
        "PYTHONHASHSEED": HASH_SEED,
        "DSBF_DB_PATH": str(workdir / "dsbf.db"),
        "DSBF_HOME": str(workdir / "home"),
    }
    command = [
        sys.executable,
        str(GOLDEN_DIR / "profile_one.py"),
        dataset.name,
        str(csv_path),
        str(out_dir),
    ]
    if dataset.time_series_column:
        command.append(dataset.time_series_column)

    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Profiling {dataset.name} exited with {completed.returncode}:\n"
            f"{completed.stderr[-4000:]}"
        )

    report = json.loads((out_dir / "report.json").read_text())
    run_info = json.loads((out_dir / "golden_run.json").read_text())
    return GoldenRun(
        report=report,
        run_info=run_info,
        roots=[str(out_dir), str(data_dir), str(workdir)],
    )
