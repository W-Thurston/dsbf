# tests/eda/test_tasks/test_detect_class_imbalance.py

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_class_imbalance import DetectClassImbalance
from tests.helpers.context_utils import make_ctx_and_task


def test_balanced_dataset_not_flagged(tmp_path) -> None:
    """A perfectly balanced binary target must not be flagged as imbalanced."""
    df = pl.DataFrame({"target": [0, 1] * 50})

    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        task_overrides={"target_column": "target", "imbalance_ratio_threshold": 0.9},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data["majority_ratio"] == 0.5
    assert result.data["is_imbalanced"] is False
    assert result.recommendations == []


def test_imbalanced_dataset_is_flagged(tmp_path) -> None:
    """A 95/5 split must be flagged as imbalanced with recommendations."""
    df = pl.DataFrame({"target": [0] * 95 + [1] * 5})

    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        task_overrides={"target_column": "target", "imbalance_ratio_threshold": 0.9},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["is_imbalanced"] is True
    assert result.data["majority_ratio"] == 0.95
    assert result.recommendations is not None
    assert any("imbalanced" in r.lower() for r in result.recommendations)


def test_skips_if_target_column_missing(tmp_path) -> None:
    """Task must return skipped status when target_column is not in the DataFrame."""
    df = pl.DataFrame({"not_target": [0, 1, 0, 1]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "skipped"
    assert "target" in result.summary["message"].lower()


def test_skips_if_no_target_column_configured(tmp_path) -> None:
    """Task must return skipped status when no target_column is configured."""
    df = pl.DataFrame({"x": [1, 2, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "skipped"


def test_empty_target_column_handled(tmp_path) -> None:
    """An empty target column must not raise an exception."""
    df = pl.DataFrame({"target": pl.Series([], dtype=pl.Int64)})

    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["majority_ratio"] == 0.0
    assert result.data["is_imbalanced"] is False


def test_single_class_target_is_imbalanced(tmp_path) -> None:
    """A target with only one class has majority_ratio 1.0 and must be flagged."""
    df = pd.DataFrame({"target": ["X"] * 10})

    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["majority_ratio"] == 1.0
    assert result.data["is_imbalanced"] is True


def test_guidance_blurbs_attached_when_imbalanced(tmp_path) -> None:
    """Both EDA and ML guidance blurbs must be attached when imbalance is detected."""
    df = pd.DataFrame({"target": ["A"] * 80 + ["B"] * 20})

    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        task_overrides={"target_column": "target", "imbalance_ratio_threshold": 0.75},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["is_imbalanced"] is True
    assert result.guidance is not None
    assert "target" in result.guidance
    assert len(result.guidance["target"]["eda"]) > 0
    assert len(result.guidance["target"]["ml"]) > 0

    # ML blurb must contain actionable steps
    ml_actions = result.guidance["target"]["ml"][0]["actions"]
    assert len(ml_actions) > 0
    action_types: set = {a["action"] for a in ml_actions}
    assert "set_class_weight" in action_types or "resample" in action_types


def test_no_guidance_when_balanced(tmp_path) -> None:
    """No guidance blurbs must be emitted for a balanced target."""
    df = pl.DataFrame({"target": [0, 1] * 50})

    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        task_overrides={"target_column": "target", "imbalance_ratio_threshold": 0.9},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["is_imbalanced"] is False
    # No guidance should be attached for a balanced target
    assert result.guidance is None or "target" not in result.guidance


def test_no_plots_generated(tmp_path) -> None:
    """Class imbalance task must not generate plots."""
    df = pl.DataFrame({"target": [0] * 90 + [1] * 10})

    ctx, task = make_ctx_and_task(
        task_cls=DetectClassImbalance,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
