# tests/eda/test_engine/test_profile_engine.py

import json
import logging
import os
import warnings

import pandas as pd
import pytest

from dsbf.core.context import AnalysisContext
from dsbf.eda.profile_engine import ProfileEngine
from dsbf.eda.task_result import TaskResult

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.network
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

    engine = ProfileEngine(config)
    engine.run()

    assert isinstance(engine.context, AnalysisContext)
    outputs = os.listdir(engine.output_dir)
    assert "report.json" in outputs

    dag_path = os.path.join(engine.output_dir, "figs", "dag.png")
    if not os.path.exists(dag_path):
        pytest.skip("DAG image not generated; possibly skipped due to environment.")


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.network
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


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.network
def test_profile_engine_runs_with_polars(clean_engine_run):
    try:
        __import__("polars")
    except ImportError:
        pytest.skip("Polars not installed - skipping test.")

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


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.network
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

    with open(os.path.join(engine.output_dir, "metadata_report.json")) as f:
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
@pytest.mark.network
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


def test_run_log_holds_task_messages_and_is_closed(tmp_path):
    data = tmp_path / "small.csv"
    data.write_text("a,b\n1,x\n2,y\n3,x\n4,y\n5,x\n")
    config = {
        "metadata": {
            "dataset_path": str(data),
            "dataset_name": "small",
            "message_verbosity": "debug",
            "profiling_depth": "basic",
        },
        "output_dir": str(tmp_path / "out"),
    }
    root = logging.getLogger("dsbf")
    handlers_before = list(root.handlers)

    engine = ProfileEngine(config)
    engine.run()

    log = (tmp_path / "out" / "run.log").read_text()
    assert "[STAGE] dsbf.core.base_engine: Starting profiling..." in log
    assert "dsbf.core.context: [" in log
    assert list(root.handlers) == handlers_before
