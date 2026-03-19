# tests/eda/test_tasks/test_categorical_length_stats.py

import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.categorical_length_stats import CategoricalLengthStats
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_categorical_length_stats_expected_output(tmp_path) -> None:
    """
    Verify string length stats are computed for categorical columns only.

    Numeric columns should be excluded via semantic type routing; text columns
    should produce mean, min, and max character length in the output data.
    """
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlotte", None],
            "city": ["New York", "Paris", "Berlin", "New York"],
            "age": [25, 30, 35, 40],  # numeric — should be excluded
        },
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
    assert result.data is not None

    # Numeric columns must not appear in output
    assert "age" not in result.data

    # city is unambiguously categorical and must be present
    assert "city" in result.data
    assert {"mean_length", "max_length", "min_length"} <= set(
        result.data["city"].keys(),
    )

    # All reported stats must be valid numeric types
    for col_stats in result.data.values():
        assert isinstance(col_stats["mean_length"], float)
        assert isinstance(col_stats["max_length"], int)
        assert isinstance(col_stats["min_length"], int)
        assert col_stats["min_length"] <= col_stats["max_length"]

    # Metadata checks
    excluded = result.metadata.get("excluded_columns", {})
    assert "age" in excluded

    column_types = result.metadata.get("column_types", {})
    assert "city" in column_types
    assert column_types["city"]["analysis_intent_dtype"] in ("categorical", "text")


def test_categorical_length_stats_no_text_columns(tmp_path) -> None:
    """Task should return empty data and no plots when no text columns are present."""
    df = pd.DataFrame(
        {
            "age": [1, 2, 3],
            "height": [5.5, 6.0, 5.8],
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=CategoricalLengthStats,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CategoricalLengthStats)

    assert result.status == "success"
    assert result.data == {}
    # Computation tasks do not generate plots — rendering is owned by
    # generate_univariate_plots and generate_dataset_summary_plots.
    assert result.plots is None
    assert result.metadata.get("column_types") is not None


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_categorical_length_stats_stat_values(tmp_path) -> None:
    """Verify stat values are numerically correct for known input."""
    df = pd.DataFrame(
        {
            "product": ["a", "bb", "ccc"],  # lengths 1, 2, 3
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=CategoricalLengthStats,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CategoricalLengthStats)

    assert result.status == "success"
    assert "product" in result.data
    stats = result.data["product"]
    assert stats["min_length"] == 1
    assert stats["max_length"] == 3
    assert abs(stats["mean_length"] - 2.0) < 1e-6


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_categorical_length_stats_nulls_excluded_from_stats(tmp_path) -> None:
    """Null values must be excluded before computing length statistics."""
    df = pd.DataFrame(
        {
            "region": ["east", None, "west", None],
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=CategoricalLengthStats,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CategoricalLengthStats)

    assert result.status == "success"
    assert "region" in result.data
    # "east" and "west" are both length 4; nulls must not inflate max
    assert result.data["region"]["min_length"] == 4
    assert result.data["region"]["max_length"] == 4


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_categorical_length_stats_excluded_columns_in_metadata(tmp_path) -> None:
    """Non-categorical columns must appear in excluded_columns metadata."""
    df = pd.DataFrame(
        {
            "label": ["a", "b", "c"],
            "score": [1.0, 2.0, 3.0],
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=CategoricalLengthStats,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CategoricalLengthStats)

    assert result.status == "success"
    excluded = result.metadata.get("excluded_columns", {})
    assert "score" in excluded
    column_types = result.metadata.get("column_types", {})
    assert "label" in column_types
    assert "score" in column_types
