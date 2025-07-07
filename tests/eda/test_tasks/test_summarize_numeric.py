# tests/eda/test_tasks/test_summarize_numeric.py

from pathlib import Path

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_numeric import SummarizeNumeric
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_numeric_expected_output(tmp_path):
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
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "a" in result.data
    assert "1%" in result.data["a"]
    assert "near_zero_variance" in result.data["b"]
    assert bool(result.data["b"]["near_zero_variance"]) is True


def test_summarize_numeric_expected_keys(tmp_path):
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
        global_overrides={"output_dir": str(tmp_path)},
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


def test_summarize_numeric_with_plots(tmp_path):
    df = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40],
            "income": [50000, 60000, 55000, 58000, 62000],
            "score": [0.8, 0.7, 0.6, 0.9, 0.85],
        }
    )

    # Use helper to create context and task
    ctx, task = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert "age" in result.data
    assert result.plots is not None
    assert "age" in result.plots

    # Check static plot path exists and clean up
    static_path = result.plots["age"]["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
    static_path.unlink()


def test_summarize_numeric_empty_dataframe(tmp_path):
    df = pd.DataFrame(columns=["a", "b", "c"])
    ctx, task = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data == {}
    assert result.plots == {}


def test_summarize_numeric_all_nulls(tmp_path):
    df = pd.DataFrame(
        {
            "x": [None, None, None],
            "y": [float("nan")] * 3,
        }
    )
    ctx, task = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data == {}
    assert result.plots == {}


def test_summarize_numeric_with_polars_input(tmp_path):
    pdf = pd.DataFrame({"score": [1.2, 2.3, 3.4], "value": [100, 150, 200]})
    pl_df = pl.from_pandas(pdf)
    ctx, task = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=pl_df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert "score" in result.data
    assert result.plots is not None
    assert "value" in result.plots
    static_path = result.plots["value"]["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
    static_path.unlink()
