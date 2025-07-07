# tests/eda/test_tasks/test_detect_target_drift.py

from pathlib import Path

import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_target_drift import DetectTargetDrift
from tests.helpers.context_utils import make_ctx_and_task


def test_numeric_target_drift_detected(tmp_path):
    current = pl.DataFrame({"target": [1.0] * 50 + [10.0] * 50})
    reference = pl.DataFrame({"target": [1.0] * 100})

    ctx, task = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
        global_overrides={"output_dir": str(tmp_path)},
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


def test_categorical_target_drift_detected(tmp_path):
    current = pl.DataFrame({"target": ["A"] * 10 + ["B"] * 90})
    reference = pl.DataFrame({"target": ["A"] * 50 + ["B"] * 50})
    ctx, task = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.data["target_type"] == "categorical"
    assert result.data["tvd"] > 0.1
    assert result.data["drift_rating"] in {"moderate", "significant"}
    assert "drift" in result.summary["message"].lower()


def test_missing_reference_skips(tmp_path):
    current = pl.DataFrame({"target": [1, 2, 3]})
    ctx, task = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "skipped"
    assert "skipped" in result.summary["message"].lower()


def test_missing_target_column_skips(tmp_path):
    current = pl.DataFrame({"x": [1, 2, 3]})
    reference = pl.DataFrame({"x": [1, 2, 3]})
    ctx, task = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "skipped"
    assert "no target column" in result.summary["message"].lower()


def test_error_handling_on_invalid_input(tmp_path):
    current = pl.DataFrame({"target": []})
    reference = pl.DataFrame({"target": [1, 2, 3]})
    ctx, task = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status in {"error", "skipped", "failed"}
    assert isinstance(result.summary, dict)
    assert any(
        kw in result.summary.get("message", "").lower()
        for kw in ["error", "failed", "skipped"]
    )


def test_numeric_target_drift_generates_plot(tmp_path):
    current = pl.DataFrame({"target": [0.0] * 50 + [5.0] * 50})
    reference = pl.DataFrame({"target": [0.0] * 100})

    ctx, task = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "target_drift" in result.plots

    plot_entry = result.plots["target_drift"]

    # Static file path
    static_path = plot_entry["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
    assert static_path.suffix == ".png"

    # Interactive structure
    interactive = plot_entry["interactive"]
    assert isinstance(interactive, dict)
    assert interactive["type"] == "histogram"
    assert "annotations" in interactive
    assert any("psi" in ann.lower() for ann in interactive["annotations"])


def test_categorical_target_drift_generates_plot(tmp_path):
    current = pl.DataFrame({"target": ["X"] * 90 + ["Y"] * 10})
    reference = pl.DataFrame({"target": ["X"] * 50 + ["Y"] * 50})

    ctx, task = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "target_drift" in result.plots

    plot_entry = result.plots["target_drift"]
    static_path = plot_entry["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()

    interactive = plot_entry["interactive"]
    assert interactive["type"] == "bar"
    assert any("tvd" in a.lower() for a in interactive["annotations"])
