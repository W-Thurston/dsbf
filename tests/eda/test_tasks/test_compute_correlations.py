# tests/eda/test_tasks/test_compute_correlations.py

import warnings
from pathlib import Path

import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.compute_correlations import ComputeCorrelations
from tests.helpers.context_utils import make_ctx_and_task


def test_compute_correlations_with_categorical(tmp_path):
    """
    Test that ComputeCorrelations detects Pearson and Cramér’s V correlations
    and returns results in expected flat dictionary format.
    """
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],  # Strong Pearson
            "cat1": ["a", "a", "b", "b", "b"],
            "cat2": ["yes", "yes", "no", "no", "no"],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeCorrelations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result is not None, "No TaskResult returned"
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    # Pearson check
    assert any("x|y" in key for key in result.data)
    assert abs(result.data["x|y"] - 1.0) < 1e-2  # Perfect correlation

    # Cramér’s V check
    assert "cat1|cat2" in result.data
    v = result.data["cat1|cat2"]
    assert isinstance(v, float)
    assert 0.0 <= v <= 1.0


def extract_warning(result: TaskResult, level: str, code: str):
    return (
        result.reliability_warnings.get(level, {}).get(code)
        if result.reliability_warnings
        else None
    )


def test_warns_on_low_row_count(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [2, 4]})  # N = 2
    ctx, task = make_ctx_and_task(
        ComputeCorrelations, df, global_overrides={"output_dir": str(tmp_path)}
    )
    result = ctx.run_task(task)

    assert extract_warning(result, "strong_warning", "low_row_count")


def test_warns_on_zero_variance(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = pd.DataFrame({"a": [1, 1, 1, 1, 1], "b": [2, 3, 4, 5, 6]})
        ctx, task = make_ctx_and_task(
            ComputeCorrelations, df, global_overrides={"output_dir": str(tmp_path)}
        )
        result = ctx.run_task(task)

        assert extract_warning(result, "strong_warning", "zero_variance")


def test_warns_on_extreme_outliers(tmp_path):
    df = pd.DataFrame({"a": [1] * 29 + [1000], "b": list(range(30))})  # z-score > 3
    ctx, task = make_ctx_and_task(
        ComputeCorrelations, df, global_overrides={"output_dir": str(tmp_path)}
    )
    result = ctx.run_task(task)

    assert extract_warning(result, "heuristic_caution", "extreme_outliers")


def test_warns_on_high_skew(tmp_path):
    df = pd.DataFrame(
        {
            "a": list(range(1, 25)) + [500, 1000, 2000, 3000, 4000, 5000],
            "b": list(range(30)),
        }
    )
    ctx, task = make_ctx_and_task(
        ComputeCorrelations, df, global_overrides={"output_dir": str(tmp_path)}
    )
    result = ctx.run_task(task)

    assert extract_warning(result, "heuristic_caution", "high_skew")


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_no_warnings_on_clean_data(tmp_path):
    df = pd.DataFrame({"a": list(range(1, 31)), "b": [x * 2 for x in range(1, 31)]})
    ctx, task = make_ctx_and_task(
        ComputeCorrelations, df, global_overrides={"output_dir": str(tmp_path)}
    )
    result = ctx.run_task(task)

    warnings = result.reliability_warnings
    assert not warnings or all(not level for level in warnings.values())


def test_warns_on_high_skew_low_n(tmp_path):
    df = pd.DataFrame({"a": [0, 0, 0, 0, 1e6], "b": [1, 2, 3, 4, 5]})  # Massive skew
    ctx, task = make_ctx_and_task(
        ComputeCorrelations, df, global_overrides={"output_dir": str(tmp_path)}
    )
    result = ctx.run_task(task)

    assert extract_warning(result, "heuristic_caution", "high_skew_low_n")


def test_warns_on_extreme_outliers_low_n(tmp_path):
    df = pd.DataFrame(
        {"a": [0, 0, 0, 0, 1e6], "b": [1, 2, 3, 4, 5]}  # Constant + 1 clear outlier
    )
    ctx, task = make_ctx_and_task(
        ComputeCorrelations, df, global_overrides={"output_dir": str(tmp_path)}
    )
    result = ctx.run_task(task)

    assert extract_warning(result, "heuristic_caution", "extreme_outliers_low_n")


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_compute_correlations_with_plots(tmp_path):
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": [2, 3, 4, 5, 6],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeCorrelations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.plots is not None
    assert "correlation_matrix" in result.plots

    static_path = result.plots["correlation_matrix"]["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
    static_path.unlink()

    interactive = result.plots["correlation_matrix"]["interactive"]
    assert isinstance(interactive, dict)
    assert interactive["type"] == "correlation"
    assert "data" in interactive
    assert isinstance(interactive["annotations"], list)


def test_compute_correlations_empty_df(tmp_path):
    df = pd.DataFrame()

    ctx, task = make_ctx_and_task(
        task_cls=ComputeCorrelations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data == {}
    assert result.plots is None or result.plots == {}


def test_compute_correlations_non_numeric_columns(tmp_path):
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["foo", "bar", "baz"]})

    ctx, task = make_ctx_and_task(
        task_cls=ComputeCorrelations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data == {}
    assert result.plots == {}
