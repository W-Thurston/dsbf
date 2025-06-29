import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_target_drift import DetectTargetDrift
from tests.helpers.context_utils import make_ctx_and_task


def test_numeric_target_drift_detected():
    current = pl.DataFrame({"target": [1.0] * 50 + [10.0] * 50})
    reference = pl.DataFrame({"target": [1.0] * 100})

    ctx, task = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["target_type"] == "numerical"
    assert result.data["psi"] > 0.1
    assert result.data["drift_rating"] in {"moderate", "significant"}
    assert result.recommendations is not None
    assert "retrain" in result.recommendations[0].lower()


def test_categorical_target_drift_detected():
    current = pl.DataFrame({"target": ["A"] * 10 + ["B"] * 90})
    reference = pl.DataFrame({"target": ["A"] * 50 + ["B"] * 50})
    ctx, task = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.data["target_type"] == "categorical"
    assert result.data["tvd"] > 0.1
    assert result.data["drift_rating"] in {"moderate", "significant"}
    assert "drift" in result.summary["message"].lower()


def test_missing_reference_skips():
    current = pl.DataFrame({"target": [1, 2, 3]})
    ctx, task = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
    )
    result = ctx.run_task(task)

    assert result.status == "skipped"
    assert "skipped" in result.summary["message"].lower()


def test_missing_target_column_skips():
    current = pl.DataFrame({"x": [1, 2, 3]})
    reference = pl.DataFrame({"x": [1, 2, 3]})
    ctx, task = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
    )
    result = ctx.run_task(task)

    assert result.status == "skipped"
    assert "no target column" in result.summary["message"].lower()


def test_error_handling_on_invalid_input():
    current = pl.DataFrame({"target": []})
    reference = pl.DataFrame({"target": [1, 2, 3]})
    ctx, task = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
    )
    result = ctx.run_task(task)

    assert result.status in {"error", "skipped", "failed"}
    assert (
        "error" in result.summary["message"].lower()
        or "skipped" in result.summary["message"].lower()
    )
