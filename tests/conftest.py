# tests/conftest.py
import json
import os
import shutil

import pytest


@pytest.fixture
def clean_engine_run(tmp_path):
    """
    Fixture to create and clean up a ProfileEngine run.
    Returns a tuple (config, cleanup_fn).
    """
    outputs_dir = "dsbf/outputs"
    previous_runs = []
    if os.path.exists("dsbf_run.json"):
        with open("dsbf_run.json", "r") as f:
            previous_runs = json.load(f)

    yield tmp_path  # Let the test run

    # Cleanup dsbf/outputs/
    for item in os.listdir(outputs_dir):
        full_path = os.path.join(outputs_dir, item)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path, ignore_errors=True)

    # Restore dsbf_run.json to previous state
    with open("dsbf_run.json", "w") as f:
        json.dump(previous_runs, f, indent=2)
