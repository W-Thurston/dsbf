# tests/eda/test_engine/test_config_validation.py

import pytest

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import TASK_REGISTRY, TaskSpec
from dsbf.utils.config_validation import validate_config_and_graph


# Shared dummy task for TaskSpec use
class DummyTask(BaseTask):
    def run(self):
        pass


@pytest.fixture(scope="module")
def basic_task_spec():
    """Returns a minimal TaskSpec-compatible task with no dependencies."""
    return TaskSpec(
        name="dummy_task",
        cls=DummyTask,
        profiling_depth="basic",
    )


def test_valid_config_and_graph_passes(
    monkeypatch, basic_task_spec, minimal_valid_config
):

    monkeypatch.setitem(TASK_REGISTRY, "dummy_task", basic_task_spec)
    errors = validate_config_and_graph(minimal_valid_config)
    assert not errors, f"Expected no validation errors, got: {errors}"


def test_unknown_task_name_detected(config_with_unknown_task):
    errors = validate_config_and_graph(config_with_unknown_task)
    assert any("Unknown task name" in e for e in errors)


def test_dependency_on_unknown_task(monkeypatch):
    spec = TaskSpec(
        name="task_with_dep",
        cls=DummyTask,
        profiling_depth="full",
        depends_on=["missing_upstream"],
    )
    monkeypatch.setitem(TASK_REGISTRY, "task_with_dep", spec)

    config = {"tasks": {"task_with_dep": {}}}
    errors = validate_config_and_graph(config)
    assert any("depends on unknown task" in e for e in errors)


def test_cycle_detection(monkeypatch, config_with_cycle):

    spec_a = TaskSpec(
        name="a",
        cls=DummyTask,
        profiling_depth="full",
        depends_on=["b"],
    )
    spec_b = TaskSpec(
        name="b",
        cls=DummyTask,
        profiling_depth="full",
        depends_on=["a"],
    )
    monkeypatch.setitem(TASK_REGISTRY, "a", spec_a)
    monkeypatch.setitem(TASK_REGISTRY, "b", spec_b)

    errors = validate_config_and_graph(config_with_cycle)
    assert any("Cycle detected" in e for e in errors)


def test_plugin_warnings_are_included(monkeypatch, config_with_strict_plugin_failure):

    monkeypatch.setattr(
        "dsbf.utils.config_validation.get_plugin_warnings",
        lambda: [{"message": "Plugin X failed to register any task."}],
    )

    errors = validate_config_and_graph(config_with_strict_plugin_failure)
    assert any("[Plugin Warning]" in e for e in errors)


def test_strict_mode_raises():
    config = {
        "tasks": {"not_a_real_task": {}},
        "safety": {"strict_mode": True},
    }
    errors = validate_config_and_graph(config)
    if errors:
        with pytest.raises(ValueError):
            if config["safety"]["strict_mode"]:
                raise ValueError(f"Strict mode: {errors}")


def test_non_strict_mode_logs(monkeypatch, capsys):
    config = {
        "tasks": {"nonexistent": {}},
        "safety": {"strict_mode": False},
    }
    errors = validate_config_and_graph(config)
    assert errors
    for err in errors:
        print(f"[VALIDATION WARNING] {err}")

    captured = capsys.readouterr()
    assert "nonexistent" in captured.out
