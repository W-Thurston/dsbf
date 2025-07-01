# tests/test_tasks/test_smoke_all_tasks.py

import polars as pl
import pytest

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_loader import load_all_tasks
from dsbf.eda.task_registry import TASK_REGISTRY, TaskSpec
from dsbf.eda.task_result import TaskResult
from dsbf.utils.task_utils import is_diagnostic_task

load_all_tasks()


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in scalar divide:RuntimeWarning"
)
def test_all_tasks_smoke(tmp_path):
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": ["x", "y", "z", "x", "y"],
            "d": [None, 2, 3, None, 5],
        }
    )
    context = AnalysisContext(df, output_dir=str(tmp_path))

    assert TASK_REGISTRY, "No tasks registered!"

    for task_name, spec in TASK_REGISTRY.items():
        assert isinstance(task_name, str), f"Registry key '{task_name}' is not a string"
        assert isinstance(spec, TaskSpec), f"{task_name} is not a TaskSpec"
        assert hasattr(spec.cls, "run"), f"{spec.cls.__name__} missing .run() method"
        assert callable(
            getattr(spec.cls(), "run", None)
        ), f"{spec.cls.__name__}.run is not callable"

        task = spec.cls()

        # Inject mock metadata for diagnostic tasks
        if is_diagnostic_task(task_name):
            context.metadata["task_durations"] = {"mock_task": 0.5}
            context.metadata["run_stats"] = {"duration": 0.5}

        result = context.run_task(task)

        assert isinstance(result, TaskResult), f"{task_name} did not return TaskResult"
        assert result.status in (
            "success",
            "skipped",
        ), f"{task_name} returned unexpected status: {result.status}"
