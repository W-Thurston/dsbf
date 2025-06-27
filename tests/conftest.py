import json
import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def clean_engine_run(tmp_path):
    """
    Fixture to create and clean up a ProfileEngine run.
    Only removes new folders added to dsbf/outputs during test.
    """
    outputs_dir = Path("dsbf/outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot existing folders before the test
    preexisting = {p.name for p in outputs_dir.iterdir() if p.is_dir()}

    # Snapshot dsbf_run.json
    dsbf_run_path = Path("dsbf_run.json")
    previous_runs = []
    if dsbf_run_path.exists():
        previous_runs = json.loads(dsbf_run_path.read_text())

    yield tmp_path  # Run the test

    # Cleanup: remove only new folders
    for p in outputs_dir.iterdir():
        if p.is_dir() and p.name not in preexisting:
            shutil.rmtree(p, ignore_errors=True)

    # Restore dsbf_run.json
    dsbf_run_path.write_text(json.dumps(previous_runs, indent=2))


@pytest.fixture(scope="session", autouse=True)
def clean_outputs_latest_after_tests():
    yield  # Let all tests run first

    latest_dir = os.path.join("dsbf", "outputs", "latest")
    if os.path.exists(latest_dir):
        for fname in os.listdir(latest_dir):
            fpath = os.path.join(latest_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath, ignore_errors=True)
    shutil.rmtree(latest_dir, ignore_errors=True)
