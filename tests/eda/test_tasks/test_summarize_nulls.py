# tests/test_tasks/test_summarize_nulls.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_nulls import SummarizeNulls
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_nulls_expanded_output():
    df = pd.DataFrame(
        {"a": [None, 1, None, 1], "b": [1, None, None, 1], "c": [1, 1, 1, 1]}
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeNulls,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "null_patterns" in result.data
    assert isinstance(result.data["null_patterns"], dict)
    assert "high_null_columns" in result.data
    assert any(col in ["a", "b"] for col in result.data["high_null_columns"])


def test_summarize_nulls_all_null_column():
    df = pd.DataFrame({"a": [None, None, None]})

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeNulls,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert "a" in result.data.get("high_null_columns", [])
