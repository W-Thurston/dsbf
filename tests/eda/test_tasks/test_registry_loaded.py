# tests/test_tasks/test_registry_loaded.py

from dsbf.eda.task_loader import load_all_tasks
from dsbf.eda.task_registry import TASK_REGISTRY


def test_task_registry_is_populated():
    # Load all tasks explicitly
    load_all_tasks()

    # Assert the registry isn't empty
    assert (
        TASK_REGISTRY
    ), "TASK_REGISTRY is empty. Did you forget to call load_all_tasks()?"

    expected = "detect_constant_columns"
    assert (
        expected in TASK_REGISTRY
    ), f"Expected task '{expected}' not found in registry."
