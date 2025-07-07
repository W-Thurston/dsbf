# tests/eda/test_tasks/test_categorical_length_stats.py

from pathlib import Path

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.categorical_length_stats import CategoricalLengthStats
from tests.helpers.context_utils import make_ctx_and_task


def test_categorical_length_stats_expected_output(tmp_path):
    """
    Test that CategoricalLengthStats correctly computes string length stats
    for text-like columns and skips numeric columns.
    """
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlotte", None],
            "city": ["New York", "Paris", "Berlin", "New York"],
            "age": [25, 30, 35, 40],  # Should be ignored
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=CategoricalLengthStats,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    # Validate TaskResult structure
    assert result is not None, "Task did not produce an output"
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert isinstance(result.data, dict)
    assert "name" in result.data
    assert "city" in result.data
    assert "age" not in result.data  # Not a string-type column

    # Validate content structure
    for col_stats in result.data.values():
        assert {"mean_length", "max_length", "min_length"} <= set(col_stats.keys())


def test_categorical_length_stats_no_text_columns(tmp_path):
    """
    Test that the task returns an empty result when no text-like columns are present.
    """
    df = pd.DataFrame(
        {
            "age": [1, 2, 3],
            "height": [5.5, 6.0, 5.8],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=CategoricalLengthStats,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data == {}
    assert result.plots is None or result.plots == {}


def test_categorical_length_stats_generates_plots(tmp_path):
    """
    Test that CategoricalLengthStats generates both static and interactive plots
    for each text-like column.
    """
    df = pd.DataFrame(
        {
            "product": ["apple", "banana", "cherry", None],
            "region": ["east", "west", "south", "east"],
            "quantity": [100, 200, 300, 400],  # Should not be plotted
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=CategoricalLengthStats,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert isinstance(result.plots, dict)
    assert set(result.plots.keys()) == {"product", "region"}

    for col in ["product", "region"]:
        plot_entry = result.plots[col]
        assert isinstance(plot_entry["static"], Path)
        assert plot_entry["static"].suffix == ".png"
        assert plot_entry["static"].exists()

        interactive = plot_entry["interactive"]
        assert isinstance(interactive, dict)
        assert interactive["type"] == "histogram"
        assert "data" in interactive
        assert "config" in interactive
        assert "annotations" in interactive
        assert "length" in interactive["config"].get("title", "").lower()
