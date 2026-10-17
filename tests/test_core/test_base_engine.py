# tests/test_core/test_base_engine.py

from pathlib import Path

import pytest

from dsbf.core.base_engine import BaseEngine
from dsbf.paths import default_outputs_dir

pytestmark = pytest.mark.unit


class DummyEngine(BaseEngine):
    def run(self):
        pass


def test_create_output_dir_creates_timestamped_folder_under_the_home():
    engine = DummyEngine(config={})
    out_dir = Path(engine._create_output_dir())
    assert out_dir.is_absolute()
    assert out_dir.parent == default_outputs_dir()
    assert out_dir.is_dir()


def test_get_git_sha_returns_str():
    engine = DummyEngine(config={})
    sha = engine._get_git_sha()
    assert isinstance(sha, str)


def test_logger_and_run_metadata_fields_present():
    config = {
        "metadata": {
            "dataset_name": "sample",
            "dataset_source": "test",
            "profiling_depth": "basic",
            "message_verbosity": "debug",
            "visualize_dag": True,
        }
    }
    engine = DummyEngine(config=config)
    md = engine.run_metadata
    assert md["engine"] == "DummyEngine"
    assert "timestamp" in md
    assert md["dataset_name"] == "sample"
    assert md["profiling_depth"] == "basic"
    assert md["message_verbosity"] == "debug"
    assert md["visualize_dag"] is True
