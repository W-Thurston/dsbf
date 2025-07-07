# tests/eda/test_tasks/test_suggest_numerical_binning.py

from pathlib import Path

import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.suggest_numerical_binning import SuggestNumericalBinning
from tests.helpers.context_utils import make_ctx_and_task


def test_suggests_log_transform_for_skewed_column(tmp_path):
    df = pl.DataFrame(
        {
            "feature": [1] * 50 + [1000] * 10,
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )

    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    suggestion = result.data["binning_suggestions"]["feature"]["suggested_binning"]
    assert "log" in suggestion


def test_suggests_equal_width_binning_when_range_is_large(tmp_path):
    df = pl.DataFrame(
        {"feature": list(range(0, 1000, 10))}  # large range, relatively uniform
    )

    ctx, task = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
        task_overrides={"skew_threshold": 10.0},  # force non-log
        global_overrides={"output_dir": str(tmp_path)},
    )

    result = ctx.run_task(task)

    assert result.data is not None
    suggestion = result.data["binning_suggestions"]["feature"]["suggested_binning"]
    assert "equal-width" in suggestion


def test_suggests_quantile_binning_when_range_is_small(tmp_path):
    df = pl.DataFrame({"feature": [10, 11, 12, 13, 14] * 20})  # small std, small range

    ctx, task = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
        task_overrides={"skew_threshold": 10.0},  # avoid log-transform
        global_overrides={"output_dir": str(tmp_path)},
    )

    result = ctx.run_task(task)

    assert result.data is not None
    suggestion = result.data["binning_suggestions"]["feature"]["suggested_binning"]
    assert "quantile" in suggestion


def test_handles_constant_column_gracefully(tmp_path):
    df = pl.DataFrame({"feature": [1] * 100})

    ctx, task = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )

    result = ctx.run_task(task)

    assert result.data is not None
    assert "feature" not in result.data["binning_suggestions"]


def test_suggest_numerical_binning_generates_plot(tmp_path):
    """
    Ensure histograms are generated for columns with binning suggestions.
    """
    df = pl.DataFrame(
        {
            "skewed": [1] * 90 + [1000] * 10,
            "uniform": list(range(100)),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )

    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "skewed" in result.plots

    plot_entry = result.plots["skewed"]
    static_path = plot_entry["static"]
    interactive = plot_entry["interactive"]

    assert isinstance(static_path, Path)
    assert static_path.exists()
    assert static_path.suffix == ".png"

    assert interactive["type"] == "histogram"
    assert "annotations" in interactive
    assert any(
        "skewness" in a.lower() or "suggested" in a.lower()
        for a in interactive["annotations"]
    )
