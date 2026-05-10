# tests/eda/test_tasks/test_detect_target_drift.py

import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_target_drift import DetectTargetDrift
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


def test_numeric_target_drift_detected(tmp_path) -> None:
    """A large shift in numeric target distribution must produce non-zero PSI."""
    current = pl.DataFrame({"target": [1.0] * 50 + [10.0] * 50})
    reference = pl.DataFrame({"target": [1.0] * 100})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectTargetDrift)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["target_type"] == "numerical"
    assert result.data["psi"] > 0.1
    assert result.data["drift_rating"] in ("moderate", "significant")
    assert result.recommendations is not None
    assert "retrain" in result.recommendations[0].lower()


def test_categorical_target_drift_detected(tmp_path) -> None:
    """A large shift in categorical target proportions must be detected."""
    current = pl.DataFrame({"target": ["A"] * 10 + ["B"] * 90})
    reference = pl.DataFrame({"target": ["A"] * 50 + ["B"] * 50})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectTargetDrift)

    assert result.status == "success"
    assert result.data["target_type"] == "categorical"
    assert result.data["tvd"] > 0.1
    assert result.data["drift_rating"] in ("moderate", "significant")
    assert "drift" in result.summary["message"].lower()


def test_skips_when_no_reference(tmp_path) -> None:
    """Task must return skipped status when no reference dataset is provided."""
    current = pl.DataFrame({"target": [1, 2, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectTargetDrift)

    assert result.status == "skipped"
    assert "skipped" in result.summary["message"].lower()


def test_skips_when_no_target_column_configured(tmp_path) -> None:
    """Task must return skipped status when target param is not configured."""
    current = pl.DataFrame({"x": [1, 2, 3]})
    reference = pl.DataFrame({"x": [1, 2, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectTargetDrift)

    assert result.status == "skipped"
    assert "no target column" in result.summary["message"].lower()


def test_handles_empty_current_series_gracefully(tmp_path) -> None:
    """
    An empty current target series must complete without raising an exception.

    scipy ks_2samp returns NaN on empty input and numpy silently produces
    NaN PSI - the task returns success with NaN/Inf metrics rather than
    failing. This is acceptable behavior; callers should not pass empty series
    in production but the task must not crash.

    """
    current = pl.DataFrame({"target": pl.Series([], dtype=pl.Float64)})
    reference = pl.DataFrame({"target": [1.0, 2.0, 3.0]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectTargetDrift)

    # Task must complete without raising - status may be success, failed, or skipped
    assert result.status in ("success", "failed", "skipped", "error")
    assert isinstance(result.summary, dict)


def test_guidance_attached_for_significant_drift(tmp_path) -> None:
    """EDA guidance blurbs must be attached when drift severity is high."""
    current = pl.DataFrame({"target": [0.0] * 50 + [10.0] * 50})
    reference = pl.DataFrame({"target": [0.0] * 100})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target", "psi": 0.1},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectTargetDrift)

    assert result.status == "success"
    assert result.data["drift_rating"] in ("moderate", "significant")
    assert result.guidance is not None
    assert "target" in result.guidance
    assert len(result.guidance["target"]["eda"]) > 0


def test_no_guidance_when_no_drift(tmp_path) -> None:
    """No guidance must be emitted when drift severity is none."""
    data: list[float] = [float(i % 5) for i in range(200)]
    current = pl.DataFrame({"target": data})
    reference = pl.DataFrame({"target": data})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectTargetDrift)

    assert result.status == "success"
    assert result.data["drift_rating"] == "none"
    assert result.guidance is None or "target" not in (result.guidance or {})


def test_no_plots_generated(tmp_path) -> None:
    """Target drift task must not generate plots."""
    current = pl.DataFrame({"target": [1.0] * 50 + [10.0] * 50})
    reference = pl.DataFrame({"target": [1.0] * 100})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectTargetDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"target": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectTargetDrift)

    assert result.status == "success"
    assert result.plots is None
