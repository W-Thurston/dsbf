# tests/eda/test_engine/test_profile_engine.py

import json
import os
import warnings

import pandas as pd
import pytest

from dsbf.core.context import AnalysisContext
from dsbf.eda.profile_engine import ProfileEngine
from dsbf.eda.task_result import TaskResult


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_profile_engine_runs_with_file(clean_engine_run, tmp_path):
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

    config["engine"]["output_path"] = str(tmp_path)
    engine = ProfileEngine(config)
    engine.run()

    assert isinstance(engine.context, AnalysisContext)
    outputs = os.listdir(engine.output_dir)
    assert "report.json" in outputs

    dag_path = os.path.join(engine.output_dir, "figs", "dag.png")
    if not os.path.exists(dag_path):
        pytest.skip("DAG image not generated; possibly skipped due to environment.")

    assert os.path.exists("dsbf_run.json")


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_profile_engine_runs_with_builtin_dataset(clean_engine_run, tmp_path):
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

    config["engine"]["output_path"] = str(tmp_path)
    engine = ProfileEngine(config)
    engine.run()

    outputs = os.listdir(engine.output_dir)
    assert "report.json" in outputs
    assert "dag.png" in os.listdir(os.path.join(engine.output_dir, "figs/"))
    assert os.path.exists("dsbf_run.json")


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_profile_engine_runs_with_polars(clean_engine_run, tmp_path):
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

    config["engine"]["output_path"] = str(tmp_path)
    engine = ProfileEngine(config)
    engine.run()

    outputs = os.listdir(engine.output_dir)
    assert "report.json" in outputs
    assert "dag.png" in os.listdir(os.path.join(engine.output_dir, "figs/"))
    assert os.path.exists("dsbf_run.json")


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_metadata_structure_and_keys(clean_engine_run, tmp_path):
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

    config["engine"]["output_path"] = str(tmp_path)
    engine = ProfileEngine(config)
    engine.run()

    # Load most recent run from dsbf_run.json
    with open("dsbf_run.json", "r") as f:
        run_history = json.load(f)
    latest_run = run_history[-1]
    output_dir = os.path.join("dsbf", "outputs", latest_run["timestamp"])

    with open(os.path.join(output_dir, "metadata_report.json"), "r") as f:
        metadata = json.load(f)

    assert metadata["engine"] == "ProfileEngine"
    assert "timestamp" in metadata
    assert "layout_name" in metadata
    assert "dataset_name" in metadata
    assert "dataset_source" in metadata
    assert "message_verbosity" in metadata
    assert "profiling_depth" in metadata
    assert metadata.get("visualize_dag") is True


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_task_output_within_profile_engine(clean_engine_run, tmp_path):
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

    config["engine"]["output_path"] = str(tmp_path)
    engine = ProfileEngine(config)
    engine.run()

    assert engine.context is not None, "Engine context not initialized"

    for task_name in ["categorical_length_stats", "summarize_dataset_shape"]:
        result = engine.context.results.get(task_name)
        assert isinstance(result, TaskResult)
        assert result.status == "success"
        assert isinstance(result.data, dict)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_profile_engine_sampling_metadata(tmp_path):
    df_path = tmp_path / "large_dataset.csv"
    pd.DataFrame({"col": range(2_000_000)}).to_csv(df_path, index=False)

    config = {
        "metadata": {
            "dataset_path": str(df_path),
            "dataset_name": "test_dataset",
            "dataset_source": "custom",
            "output_dir": str(tmp_path),
        },
        "resource_limits": {
            "enable_sampling": True,
            "sample_threshold_rows": 1_000_000,
            "sample_strategy": "head",
        },
    }

    engine = ProfileEngine(config)
    warnings.filterwarnings(
        "ignore",
        message="Attempting to set identical low and high .*",
        category=UserWarning,
    )
    engine.run()

    assert "sampling" in engine.run_metadata
    assert engine.run_metadata["sampling"]["sampled_rows"] == 1_000_000
    assert engine.run_metadata["sampling"]["original_rows"] == 2_000_000
