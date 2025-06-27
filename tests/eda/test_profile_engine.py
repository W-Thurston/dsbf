import json
import os

import pytest

from dsbf.eda.profile_engine import ProfileEngine
from dsbf.eda.task_result import TaskResult


def test_profile_engine_runs_with_file(clean_engine_run):
    tmp_path = clean_engine_run
    config = {
        "engine": "ProfileEngine",
        "output_format": ["json", "md"],
        "visualize_dag": True,
    }
    sample_path = tmp_path / "sample_data.csv"
    sample_path.write_text("a,b\n1,2\n3,4")
    config["dataset"] = str(sample_path)

    engine = ProfileEngine(config)
    engine.run()

    outputs = os.listdir(engine.output_dir)
    assert "report.json" in outputs
    assert "dag.png" in os.listdir(os.path.join(engine.output_dir, "figs/"))
    assert os.path.exists("dsbf_run.json")


def test_profile_engine_runs_with_builtin_dataset(clean_engine_run):
    config = {
        "engine": "ProfileEngine",
        "dataset_name": "iris",
        "dataset_source": "sklearn",
        "output_format": ["json", "md"],
        "visualize_dag": True,
    }

    engine = ProfileEngine(config)
    engine.run()

    outputs = os.listdir(engine.output_dir)
    assert "report.json" in outputs
    assert "dag.png" in os.listdir(os.path.join(engine.output_dir, "figs/"))
    assert os.path.exists("dsbf_run.json")


def test_profile_engine_runs_with_polars(clean_engine_run):
    try:
        __import__("polars")
    except ImportError:
        pytest.skip("Polars not installed — skipping test.")

    config = {
        "engine": "ProfileEngine",
        "dataset_name": "iris",
        "dataset_source": "sklearn",
        "output_format": ["json", "md"],
        "backend": "polars",
        "visualize_dag": True,
    }

    engine = ProfileEngine(config)
    engine.run()

    outputs = os.listdir(engine.output_dir)
    assert "report.json" in outputs
    assert "dag.png" in os.listdir(os.path.join(engine.output_dir, "figs/"))
    assert os.path.exists("dsbf_run.json")


def test_metadata_structure_and_keys(clean_engine_run):
    config = {
        "engine": "ProfileEngine",
        "dataset_name": "iris",
        "dataset_source": "sklearn",
        "output_format": ["json", "md"],
        "backend": "polars",
        "message_verbosity": "debug",
        "profiling_depth": "full",
        "visualize_dag": True,
        "layout_name": "default",
    }

    engine = ProfileEngine(config)
    engine.run()

    with open("dsbf_run.json", "r") as f:
        run_history = json.load(f)
    latest_run = run_history[-1]
    output_dir = os.path.join("dsbf", "outputs", latest_run["timestamp"])

    with open(os.path.join(output_dir, "report.json"), "r") as f:
        report = json.load(f)

    metadata = report.get("metadata", {})
    assert "engine" in metadata
    assert "timestamp" in metadata
    assert "inferred_stage" in metadata
    assert "layout_name" in metadata
    assert "config" in metadata
    assert isinstance(metadata["config"], dict)

    required_config_keys = [
        "engine",
        "dataset_name",
        "dataset_source",
        "backend",
    ]

    for key in required_config_keys:
        assert (
            key in metadata["config"] or key in metadata
        ), f"Missing key in config: {key}"


def test_task_output_within_profile_engine(clean_engine_run):
    """
    Sanity test to confirm that a known task produces results through AnalysisContext.
    """
    config = {
        "engine": "ProfileEngine",
        "dataset_name": "iris",
        "dataset_source": "sklearn",
        "output_format": ["json"],
    }

    engine = ProfileEngine(config)
    engine.run()

    # CategoricalLengthStats should always run
    assert engine.context is not None, "Engine context not initialized"
    result = engine.context.results.get("CategoricalLengthStats")
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert isinstance(result.data, dict)
