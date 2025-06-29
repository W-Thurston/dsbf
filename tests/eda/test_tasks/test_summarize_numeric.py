# tests/test_tasks/test_summarize_numeric.py

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_numeric import SummarizeNumeric
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_numeric_expected_output():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 10, 10, 10, 10],  # zero variance
            "c": [100, 200, 300, 400, 500],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "a" in result.data
    assert "1%" in result.data["a"]
    assert "near_zero_variance" in result.data["b"]
    assert bool(result.data["b"]["near_zero_variance"]) is True


def test_summarize_numeric_expected_keys():
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 5, 5, 5, 5],  # zero variance
            "c": ["x", "y", "z", "x", "y"],  # non-numeric
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    expected_keys = {
        "count",
        "mean",
        "std",
        "min",
        "1%",
        "5%",
        "25%",
        "50%",
        "75%",
        "95%",
        "99%",
        "max",
        "near_zero_variance",
    }

    for col, stats in result.data.items():
        assert isinstance(stats, dict), f"Stats for column {col} not a dict"
        assert expected_keys.issubset(
            stats.keys()
        ), f"Missing keys in summary for column {col}: {expected_keys - stats.keys()}"
