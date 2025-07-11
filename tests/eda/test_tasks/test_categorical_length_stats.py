# tests/eda/test_tasks/test_categorical_length_stats.py

from pathlib import Path

import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.categorical_length_stats import CategoricalLengthStats
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_categorical_length_stats_expected_output(tmp_path):
    """
    Test that CategoricalLengthStats correctly computes string length stats
    for text-like columns and skips numeric columns.
    """
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlotte", None],
            "city": ["New York", "Paris", "Berlin", "New York"],
            "age": [25, 30, 35, 40],  # Should be excluded
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=CategoricalLengthStats,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CategoricalLengthStats)

    assert result is not None
    assert isinstance(result, TaskResult)
    assert result.status == "success"

    # Core output checks
    assert result.data is not None
    assert len(result.data) >= 1
    assert all("mean_length" in v for v in result.data.values())
    assert "city" in result.data
    assert "age" not in result.data

    for col_stats in result.data.values():
        assert {"mean_length", "max_length", "min_length"} <= set(col_stats.keys())

    # Metadata checks
    excluded = result.metadata.get("excluded_columns", {})
    assert "age" in excluded

    column_types = result.metadata.get("column_types", {})
    assert "name" in column_types
    assert column_types["name"]["analysis_intent_dtype"] in [
        "categorical",
        "text",
        "id",
    ]
    assert column_types["name"]["inferred_dtype"] in ["object", "str"]
    if column_types["name"]["analysis_intent_dtype"] == "id":
        assert "name" not in result.data


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

    ctx, _ = make_ctx_and_task(
        task_cls=CategoricalLengthStats,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CategoricalLengthStats)

    assert result.status == "success"
    assert result.data == {}
    assert result.plots == {}
    assert result.metadata.get("column_types") is not None


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_categorical_length_stats_generates_plots(tmp_path):
    """
    Test that CategoricalLengthStats generates both static and interactive plots
    for each text-like column.
    """
    df = pd.DataFrame(
        {
            "product": ["apple", "banana", "cherry", None],
            "region": ["east", "west", "south", "east"],
            "quantity": [100, 200, 300, 400],
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=CategoricalLengthStats,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CategoricalLengthStats)

    assert result.status == "success"
    assert result.plots is not None
    assert set(result.plots.keys()).issubset({"product", "region"})
    assert len(result.plots) >= 1

    for col, plot_entry in result.plots.items():
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
