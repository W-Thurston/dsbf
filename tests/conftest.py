import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from dsbf.config import load_default_config
from dsbf.eda.profile_engine import ProfileEngine


@pytest.fixture
def clean_engine_run(tmp_path):
    outputs_dir = Path("dsbf/outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    preexisting: set[str] = {p.name for p in outputs_dir.iterdir() if p.is_dir()}

    dsbf_run_path = Path("dsbf_run.json")
    previous_runs: list = []
    if dsbf_run_path.exists():
        previous_runs = json.loads(dsbf_run_path.read_text())

    def run_engine_with_temp_config(extra_config=None) -> Path:
        config: dict[str, Any] = load_default_config()
        config.setdefault("engine", {})["output_path"] = str(tmp_path)
        # Override the sentinel dataset name with a real seaborn dataset.
        # default_config.yaml uses "default_dataset_name" deliberately so
        # callers that forget to set a dataset fail loudly — conftest must
        # always supply a real one.
        config.setdefault("metadata", {})["dataset_name"] = "titanic"
        config.setdefault("metadata", {})["dataset_source"] = "seaborn"
        if extra_config:
            config.update(extra_config)

        engine = ProfileEngine(config=config)
        engine.run()

        report_path: Path = Path(engine.output_dir) / "report.json"
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

    latest_dir: str = os.path.join("dsbf", "outputs", "latest")
    if os.path.exists(latest_dir):
        for fname in os.listdir(latest_dir):
            fpath: str = os.path.join(latest_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath, ignore_errors=True)
    shutil.rmtree(latest_dir, ignore_errors=True)


@pytest.fixture
def minimal_valid_config() -> dict[
    str, dict[str, bool] | dict[str, dict] | dict[str, str]
]:
    """A minimal config with one valid task and strict mode on."""
    return {
        "tasks": {"dummy_task": {}},
        "safety": {"strict_mode": True},
        "metadata": {"profiling_depth": "basic"},
    }


@pytest.fixture
def config_with_unknown_task() -> dict[str, dict[str, bool] | dict[str, dict]]:
    """Includes a task that does not exist in the registry."""
    return {
        "tasks": {"not_a_real_task": {}},
        "safety": {"strict_mode": False},
    }


@pytest.fixture
def config_with_cycle() -> dict[str, dict[str, bool] | dict[str, dict]]:
    """
    Tasks 'a' and 'b' will be registered in the test with cyclic deps.
    This config sets strict mode to true to test validation error.
    """
    return {
        "tasks": {"a": {}, "b": {}},
        "safety": {"strict_mode": True},
    }


@pytest.fixture
def config_with_schema_validation() -> dict[
    str, dict[str, bool | dict[str, list] | str] | dict[str, dict] | dict[str, str]
]:
    """Enables schema validation with one required column."""
    return {
        "schema_validation": {
            "enable_schema_validation": True,
            "fail_or_warn": "warn",
            "schema": {"required_columns": ["A"]},
        },
        "tasks": {"schema_validation": {}},
        "metadata": {"dataset_name": "custom"},
    }


@pytest.fixture
def config_with_strict_plugin_failure() -> dict[str, dict[str, bool] | dict]:
    """Tests strict-mode behavior when plugin warnings are present."""
    return {
        "tasks": {},
        "safety": {"strict_mode": True},
    }
