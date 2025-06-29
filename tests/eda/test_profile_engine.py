import json
import os

import pytest

from dsbf.core.context import AnalysisContext
from dsbf.eda.profile_engine import ProfileEngine
from dsbf.eda.task_result import TaskResult


def test_profile_engine_runs_with_file(clean_engine_run):
    tmp_path = clean_engine_run
    config = {
        "metadata": {
            "dataset_name": "titanic",
            "dataset_source": "seaborn",
            "message_verbosity": "debug",  # quiet | info | debug
            "profiling_depth": "full",  # basic | standard | full
            "output_format": ["md", "json"],
            "visualize_dag": True,
            "layout_name": "default",
        },
        "engine": {
            "engine": "ProfileEngine",
            "backend": "polars",  # pandas | polars
            "reference_dataset_path": None,  # default: disabled unless user sets it
        },
    }
    sample_path = tmp_path / "sample_data.csv"
    sample_path.write_text("a,b\n1,2\n3,4")
    config["metadata"]["dataset"] = str(sample_path)

    engine = ProfileEngine(config)
    engine.run()

    assert isinstance(engine.context, AnalysisContext)
    outputs = os.listdir(engine.output_dir)
    assert "report.json" in outputs

    dag_path = os.path.join(engine.output_dir, "figs", "dag.png")
    if not os.path.exists(dag_path):
        pytest.skip("DAG image not generated; possibly skipped due to environment.")

    assert os.path.exists("dsbf_run.json")


def test_profile_engine_runs_with_builtin_dataset(clean_engine_run):
    config = {
        "metadata": {
            "dataset_name": "titanic",
            "dataset_source": "seaborn",
            "message_verbosity": "debug",  # quiet | info | debug
            "profiling_depth": "full",  # basic | standard | full
            "output_format": ["md", "json"],
            "visualize_dag": True,
            "layout_name": "default",
        },
        "engine": {
            "engine": "ProfileEngine",
            "backend": "polars",  # pandas | polars
            "reference_dataset_path": None,  # default: disabled unless user sets it
        },
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
        "metadata": {
            "dataset_name": "titanic",
            "dataset_source": "seaborn",
            "message_verbosity": "debug",  # quiet | info | debug
            "profiling_depth": "full",  # basic | standard | full
            "output_format": ["md", "json"],
            "visualize_dag": True,
            "layout_name": "default",
        },
        "engine": {
            "engine": "ProfileEngine",
            "backend": "polars",  # pandas | polars
            "reference_dataset_path": None,  # default: disabled unless user sets it
        },
    }

    engine = ProfileEngine(config)
    engine.run()

    outputs = os.listdir(engine.output_dir)
    assert "report.json" in outputs
    assert "dag.png" in os.listdir(os.path.join(engine.output_dir, "figs/"))
    assert os.path.exists("dsbf_run.json")


def test_metadata_structure_and_keys(clean_engine_run):
    config = {
        "metadata": {
            "dataset_name": "titanic",
            "dataset_source": "seaborn",
            "message_verbosity": "debug",
            "profiling_depth": "full",
            "output_format": ["md", "json"],
            "visualize_dag": True,
            "layout_name": "default",
        },
        "engine": {
            "engine": "ProfileEngine",
            "backend": "polars",
            "reference_dataset_path": None,
        },
    }

    engine = ProfileEngine(config)
    engine.run()

    # Load most recent run from dsbf_run.json
    with open("dsbf_run.json", "r") as f:
        run_history = json.load(f)
    latest_run = run_history[-1]
    output_dir = os.path.join("dsbf", "outputs", latest_run["timestamp"])

    with open(os.path.join(output_dir, "report.json"), "r") as f:
        report = json.load(f)

    metadata = report.get("metadata", {})
    assert metadata["engine"] == "ProfileEngine"
    assert "timestamp" in metadata
    assert "layout_name" in metadata
    assert "dataset_name" in metadata
    assert "dataset_source" in metadata
    assert "message_verbosity" in metadata
    assert "profiling_depth" in metadata
    assert metadata.get("visualize_dag") is True

    config_block = report.get("config", {})
    assert "engine" in config_block


def test_task_output_within_profile_engine(clean_engine_run):
    config = {
        "metadata": {
            "dataset_name": "titanic",
            "dataset_source": "seaborn",
            "message_verbosity": "debug",  # quiet | info | debug
            "profiling_depth": "full",  # basic | standard | full
            "output_format": ["md", "json"],
            "visualize_dag": True,
            "layout_name": "default",
        },
        "engine": {
            "engine": "ProfileEngine",
            "backend": "polars",  # pandas | polars
            "reference_dataset_path": None,  # default: disabled unless user sets it
        },
    }

    engine = ProfileEngine(config)
    engine.run()

    assert engine.context is not None, "Engine context not initialized"

    for task_name in ["categorical_length_stats", "summarize_dataset_shape"]:
        result = engine.context.results.get(task_name)
        assert isinstance(result, TaskResult)
        assert result.status == "success"
        assert isinstance(result.data, dict)
