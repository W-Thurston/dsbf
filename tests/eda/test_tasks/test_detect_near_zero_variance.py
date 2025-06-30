# tests/eda/test_tasks/test_detect_near_zero_variance.py

import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_near_zero_variance import DetectNearZeroVariance
from tests.helpers.context_utils import make_ctx_and_task


def test_detects_near_zero_variance():
    df = pl.DataFrame(
        {
            "constant": [3.14] * 100,
            "low_var": [1.00001 + (i % 2) * 0.00001 for i in range(100)],
            "normal": list(range(100)),
        }
    )
    ctx, task = make_ctx_and_task(
        task_cls=DetectNearZeroVariance,
        current_df=df,
        task_overrides={"threshold": 1e-4},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "constant" in result.data["low_variance_columns"]
    assert "low_var" in result.data["low_variance_columns"]
    assert "normal" not in result.data["low_variance_columns"]


def test_skips_non_numeric():
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "category": ["yes", "no", "yes", "no"],
        }
    )
    ctx, task = make_ctx_and_task(
        task_cls=DetectNearZeroVariance,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.data == {"low_variance_columns": {}}
    assert result.recommendations is not None
    assert len(result.recommendations) == 0
