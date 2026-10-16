# tests/eda/test_engine/test_default_config.py

import copy

import numpy as np
import pandas as pd
import pytest

from dsbf.config import load_default_config
from dsbf.core.base_task import BaseTask
from dsbf.core.context import AnalysisContext
from dsbf.eda.profile_engine import ProfileEngine
from dsbf.eda.task_registry import load_task_group
from dsbf.utils.config_validation import validate_config_and_graph


class _SharedParamTask(BaseTask):
    def run(self) -> None:
        pass


def _load_default_task_groups(config: dict) -> None:
    for group in config.get("task_groups", ["core"]):
        load_task_group(group)


@pytest.mark.unit
def test_default_config_passes_validation():
    config = load_default_config()
    _load_default_task_groups(config)

    assert validate_config_and_graph(config) == []


@pytest.mark.unit
def test_shared_time_series_settings_are_not_a_task_entry():
    config = load_default_config()

    assert "time_series" in config
    assert "time_series" not in config["tasks"]


@pytest.mark.unit
def test_time_series_block_left_under_tasks_is_flagged():
    config = copy.deepcopy(load_default_config())
    _load_default_task_groups(config)
    config["tasks"]["time_series"] = config["time_series"]

    errors = validate_config_and_graph(config)

    assert errors == ["Unknown task name in config['tasks']: 'time_series'"]


@pytest.mark.unit
def test_shared_param_reads_top_level_block_only():
    ctx = AnalysisContext(
        data=pd.DataFrame({"x": [1]}),
        config={
            "time_series": {"enabled": True, "frequency": "W"},
            "tasks": {"time_series": {"enabled": False, "frequency": "D"}},
        },
    )
    task = _SharedParamTask()
    task.context = ctx

    assert task.get_shared_param("time_series", "enabled") is True
    assert task.get_shared_param("time_series", "frequency") == "W"
    assert task.get_shared_param("time_series", "missing", "fallback") == "fallback"
    assert task.get_shared_param("absent_block", "enabled", False) is False


@pytest.mark.integration
def test_strict_mode_run_with_default_config_completes(tmp_path):
    rng = np.random.default_rng(0)
    csv_path = tmp_path / "strict.csv"
    pd.DataFrame(
        {
            "amount": rng.normal(100, 15, 200),
            "group": rng.choice(["a", "b", "c"], 200),
            "flag": rng.integers(0, 2, 200),
        }
    ).to_csv(csv_path, index=False)

    config = load_default_config()
    config["safety"]["strict_mode"] = True
    config["metadata"]["dataset_path"] = str(csv_path)
    config["metadata"]["profiling_depth"] = "basic"
    config["metadata"]["visualize_dag"] = False
    config["output_dir"] = str(tmp_path / "out")

    engine = ProfileEngine(config=config)
    engine.run()

    assert (tmp_path / "out" / "report.json").exists()
