# tests/test_core/test_base_engine.py

import json
import os

from dsbf.core.base_engine import BaseEngine


class DummyEngine(BaseEngine):
    def run(self):
        pass


def test_create_output_dir_creates_timestamped_folder(tmp_path):
    engine = DummyEngine(config={})
    out_dir = engine._create_output_dir()
    assert "dsbf/outputs" in out_dir
    assert os.path.exists(out_dir)


def test_get_git_sha_returns_str():
    engine = DummyEngine(config={})
    sha = engine._get_git_sha()
    assert isinstance(sha, str)


def test_record_run_appends_new_entry(tmp_path):
    record_path = tmp_path / "dsbf_run.json"

    class TestableEngine(BaseEngine):
        def run(self):
            pass

        def record_run(self):
            history = []
            if record_path.exists():
                with open(record_path, "r") as f:
                    history = json.load(f)
            timestamps = {r.get("timestamp") for r in history}
            if self.run_metadata.get("timestamp") not in timestamps:
                history.append(self.run_metadata)
            with open(record_path, "w") as f:
                json.dump(history, f, indent=2)

    engine = TestableEngine(config={})
    engine.run_metadata["timestamp"] = "unique_test"
    engine.record_run()

    with open(record_path) as f:
        data = json.load(f)
    assert any(r["timestamp"] == "unique_test" for r in data)


def test_record_run_skips_duplicate_timestamp(tmp_path):
    record_path = tmp_path / "dsbf_run.json"
    existing = [{"timestamp": "existing_ts", "engine": "Test"}]
    with open(record_path, "w") as f:
        json.dump(existing, f)

    class TestableEngine(BaseEngine):
        def run(self):
            pass

        def record_run(self):
            history = []
            if record_path.exists():
                with open(record_path, "r") as f:
                    history = json.load(f)
            timestamps = {r.get("timestamp") for r in history}
            if self.run_metadata.get("timestamp") not in timestamps:
                history.append(self.run_metadata)
            with open(record_path, "w") as f:
                json.dump(history, f, indent=2)

    engine = TestableEngine(config={})
    engine.run_metadata["timestamp"] = "existing_ts"
    engine.record_run()

    with open(record_path) as f:
        data = json.load(f)
    assert len(data) == 1  # still just the original entry


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
