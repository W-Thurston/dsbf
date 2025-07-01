# tests/test_runtime_metadata.py

import json
import os

import pytest

from dsbf.config import load_default_config
from dsbf.eda.profile_engine import ProfileEngine


@pytest.fixture
def temp_output_dir(tmp_path):
    cfg = load_default_config()
    cfg["metadata"]["dataset_name"] = "iris"
    cfg["metadata"]["profiling_depth"] = "basic"
    cfg["metadata"]["message_verbosity"] = "quiet"  # Avoid clutter
    cfg["metadata"]["visualize_dag"] = False

    engine = ProfileEngine(cfg)
    engine.run()

    return engine.output_dir


def test_metadata_file_exists(temp_output_dir):
    metadata_path = os.path.join(temp_output_dir, "metadata_report.json")
    assert os.path.exists(metadata_path), "metadata_report.json was not created."


def test_metadata_fields_present(temp_output_dir):
    path = os.path.join(temp_output_dir, "metadata_report.json")
    with open(path, "r") as f:
        metadata = json.load(f)

    required_keys = [
        "timestamp",
        "engine",
        "dsbf_version",
        "git_sha",
        "host",
        "os",
        "python",
        "run_stats",
        "task_durations",
    ]
    for key in required_keys:
        assert key in metadata, f"Missing key in metadata_report: {key}"

    # Check run_stats structure
    stats = metadata["run_stats"]
    assert "start_time" in stats and "end_time" in stats, "run_stats missing fields"
    assert isinstance(stats["total_tasks"], int)

    # Check task durations
    durations = metadata["task_durations"]
    assert isinstance(durations, dict)
    for task, duration in durations.items():
        assert isinstance(duration, float)


def test_report_and_metadata_are_separate(temp_output_dir):
    report_path = os.path.join(temp_output_dir, "report.json")
    metadata_path = os.path.join(temp_output_dir, "metadata_report.json")

    with open(report_path, "r") as f1, open(metadata_path, "r") as f2:
        report = json.load(f1)
        metadata = json.load(f2)

    assert "results" in report
    assert "results" not in metadata
    assert "run_stats" in metadata
