# tests/eda/test_tasks/test_summarize_value_counts.py

from pathlib import Path

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_value_counts import SummarizeValueCounts
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_value_counts_expected_output(tmp_path):
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "a", "c", "b", "c", "c", "c"],
            "num": [1, 2, 1, 3, 2, 2, 1, 3, 3],
            "misc": [None, None, "x", "x", "x", "y", "y", "y", "y"],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeValueCounts,
        current_df=df,
        task_overrides={"top_k": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "cat" in result.data
    assert isinstance(result.data["cat"], dict)
    assert len(result.data["cat"]) <= 2
    assert "a" in result.data["cat"] or "c" in result.data["cat"]


def test_summarize_value_counts_high_cardinality(tmp_path):
    df = pd.DataFrame({"id": [f"id_{i}" for i in range(100)]})

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeValueCounts,
        current_df=df,
        task_overrides={"top_k": 5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert "id" in result.data
    assert len(result.data["id"]) <= 5


def test_summarize_value_counts_with_plots(tmp_path):
    df = pd.DataFrame(
        {
            "fruit": ["apple", "banana", "apple", "orange", "apple"],
            "color": ["red", "yellow", "green", "orange", "red"],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeValueCounts,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
        task_overrides={"top_k": 3},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "fruit" in result.plots

    # Static plot file exists
    static_path = result.plots["fruit"]["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
    static_path.unlink()

    # Interactive has annotation
    interactive = result.plots["fruit"]["interactive"]
    assert isinstance(interactive, dict)
    assert "annotations" in interactive
    assert any("Top value" in a for a in interactive["annotations"])


def test_summarize_value_counts_constant_column(tmp_path):
    df = pd.DataFrame({"col": ["same"] * 5})
    ctx, task = make_ctx_and_task(
        task_cls=SummarizeValueCounts,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
        task_overrides={"top_k": 1},
    )
    result = ctx.run_task(task)
    assert result.data is not None
    assert "col" in result.data
    assert result.plots is not None
    assert result.plots.get("col") is not None  # Still produces a barplot


def test_summarize_value_counts_empty_df(tmp_path):
    df = pd.DataFrame()
    ctx, task = make_ctx_and_task(
        task_cls=SummarizeValueCounts,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)
    assert result.status == "success"
    assert result.data == {}
    assert result.plots == {}
