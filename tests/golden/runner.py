"""Run DSBF on one dataset in an isolated child process and load what it wrote."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from tests.golden.datasets import GOLDEN_DIR, GoldenDataset

RUN_TIMEOUT_SECONDS = 1800


def profile(dataset: GoldenDataset, workdir: Path) -> dict[str, Any]:
    """The ``Run`` of the core pack on ``dataset``, as a dict.

    The child runs under ``workdir`` with its own ``DSBF_HOME``, so nothing it writes
    reaches the developer's directory, and the run is read from this child's output
    directory, never found by searching.
    """
    workdir = workdir.resolve()
    data_dir = workdir / "data"
    out_dir = workdir / "out"
    cwd = workdir / "cwd"
    for directory in (data_dir, out_dir, cwd):
        directory.mkdir(parents=True, exist_ok=True)

    csv_path = dataset.write_csv(data_dir).resolve()
    env = {**os.environ, "DSBF_HOME": str(workdir / "home")}
    if dataset.core_params:
        env["DSBF_GOLDEN_CORE_PARAMS"] = json.dumps(dataset.core_params)
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
    return json.loads((out_dir / "run.json").read_text())
