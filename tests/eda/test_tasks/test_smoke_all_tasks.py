# tests/test_tasks/test_smoke_all_tasks.py

import polars as pl
import pytest

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_registry import TASK_REGISTRY
from dsbf.eda.task_result import TaskResult


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in scalar divide:RuntimeWarning"
)
def test_all_tasks_smoke():
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": ["x", "y", "z", "x", "y"],
            "d": [None, 2, 3, None, 5],
        }
    )

    context = AnalysisContext(df)

    for task_name, task_cls in TASK_REGISTRY.items():
        task = task_cls()
        result = context.run_task(task)

        assert isinstance(result, TaskResult), f"{task_name} did not return TaskResult"
        assert result.status in (
            "success",
            "skipped",
        ), f"{task_name} status: {result.status}"
