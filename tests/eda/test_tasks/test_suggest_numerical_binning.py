# tests/eda/test_tasks/test_suggest_numerical_binning.py

import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.suggest_numerical_binning import SuggestNumericalBinning
from tests.helpers.context_utils import make_ctx_and_task


def test_suggests_log_transform_for_skewed_column():
    df = pl.DataFrame(
        {
            "feature": [1] * 50 + [1000] * 10,
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
    )

    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    suggestion = result.data["binning_suggestions"]["feature"]["suggested_binning"]
    assert "log" in suggestion


def test_suggests_equal_width_binning_when_range_is_large():
    df = pl.DataFrame(
        {"feature": list(range(0, 1000, 10))}  # large range, relatively uniform
    )

    ctx, task = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
        task_overrides={"skew_threshold": 10.0},  # force non-log
    )

    result = ctx.run_task(task)

    assert result.data is not None
    suggestion = result.data["binning_suggestions"]["feature"]["suggested_binning"]
    assert "equal-width" in suggestion


def test_suggests_quantile_binning_when_range_is_small():
    df = pl.DataFrame({"feature": [10, 11, 12, 13, 14] * 20})  # small std, small range

    ctx, task = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
        task_overrides={"skew_threshold": 10.0},  # avoid log-transform
    )

    result = ctx.run_task(task)

    assert result.data is not None
    suggestion = result.data["binning_suggestions"]["feature"]["suggested_binning"]
    assert "quantile" in suggestion


def test_handles_constant_column_gracefully():
    df = pl.DataFrame({"feature": [1] * 100})

    ctx, task = make_ctx_and_task(
        task_cls=SuggestNumericalBinning,
        current_df=df,
    )

    result = ctx.run_task(task)

    assert result.data is not None
    assert "feature" not in result.data["binning_suggestions"]
