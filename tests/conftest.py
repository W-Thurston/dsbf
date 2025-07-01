import json
import os
import shutil
from pathlib import Path

import pytest

from dsbf.config import load_default_config
from dsbf.eda.profile_engine import ProfileEngine


@pytest.fixture
def clean_engine_run(tmp_path):
    outputs_dir = Path("dsbf/outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    preexisting = {p.name for p in outputs_dir.iterdir() if p.is_dir()}

    dsbf_run_path = Path("dsbf_run.json")
    previous_runs = []
    if dsbf_run_path.exists():
        previous_runs = json.loads(dsbf_run_path.read_text())

    def run_engine_with_temp_config(extra_config=None):
        config = load_default_config()
        config.setdefault("engine", {})["output_path"] = str(
            tmp_path
        )  # ✅ fixed location
        if extra_config:
            config.update(extra_config)

        engine = ProfileEngine(config=config)
        engine.run()

        report_path = Path(engine.output_dir) / "report.json"
        assert report_path.exists(), f"Report not found at {report_path}"
        return report_path

    yield run_engine_with_temp_config

    for p in outputs_dir.iterdir():
        if p.is_dir() and p.name not in preexisting:
            shutil.rmtree(p, ignore_errors=True)

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
