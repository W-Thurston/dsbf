# tests/eda/test_tasks/test_detect_class_imbalance.py
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_class_imbalance import DetectClassImbalance
from tests.helpers.context_utils import make_ctx_and_task


def test_class_imbalance_balanced():
    df = pl.DataFrame({"target": [0, 1] * 50})
    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        task_overrides={"target_column": "target", "imbalance_ratio_threshold": 0.9},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["majority_ratio"] == 0.5
    assert result.data["is_imbalanced"] is False
    assert result.recommendations == []


def test_class_imbalance_detects_warning():
    df = pl.DataFrame({"target": [0] * 95 + [1] * 5})
    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        task_overrides={"target_column": "target", "imbalance_ratio_threshold": 0.9},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.data["is_imbalanced"] is True
    assert result.data["majority_ratio"] == 0.95
    assert result.recommendations is not None
    assert any("imbalanced" in r.lower() for r in result.recommendations)


def test_class_imbalance_skips_if_missing_target():
    df = pl.DataFrame({"not_target": [0, 1, 0, 1]})
    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        task_overrides={"target_column": "target"},
    )
    result = ctx.run_task(task)

    assert result.status == "skipped"
    assert "target" in result.summary["message"].lower()


def test_class_imbalance_handles_empty_df():
    df = pl.DataFrame({"target": []})
    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        task_overrides={"target_column": "target"},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.data["majority_ratio"] == 0.0
    assert result.data["is_imbalanced"] is False
