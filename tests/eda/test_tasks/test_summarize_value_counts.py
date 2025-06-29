# tests/test_tasks/test_summarize_value_counts.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_value_counts import SummarizeValueCounts
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_value_counts_expected_output():
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "a", "c", "b", "c", "c", "c"],
            "num": [1, 2, 1, 3, 2, 2, 1, 3, 3],
            "misc": [None, None, "x", "x", "x", "y", "y", "y", "y"],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeValueCounts, current_df=df, task_overrides={"top_k": 2}
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "cat" in result.data
    assert isinstance(result.data["cat"], dict)
    assert len(result.data["cat"]) <= 2
    assert "a" in result.data["cat"] or "c" in result.data["cat"]


def test_summarize_value_counts_high_cardinality():
    df = pd.DataFrame({"id": [f"id_{i}" for i in range(100)]})

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeValueCounts, current_df=df, task_overrides={"top_k": 5}
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert "id" in result.data
    assert len(result.data["id"]) <= 5
