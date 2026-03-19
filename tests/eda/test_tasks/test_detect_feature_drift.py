# tests/eda/test_tasks/test_detect_feature_drift.py

import numpy as np
import polars as pl
from _collections_abc import Generator

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_feature_drift import DetectFeatureDrift
from tests.helpers.context_utils import make_ctx_and_task


def test_numeric_drift_detected(tmp_path):
    """A large shift in numeric distribution must produce high/moderate severity."""
    rng: Generator = np.random.default_rng(42)
    reference = pl.DataFrame({"x": rng.normal(0, 1, 1000).tolist()})
    current = pl.DataFrame({"x": rng.normal(3, 1, 1000).tolist()})

    ctx, task = make_ctx_and_task(
        task_cls=DetectFeatureDrift,
        current_df=current,
        reference_df=reference,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "x" in result.data
    assert result.data["x"]["type"] == "numerical"
    assert result.data["x"]["severity"] in ("moderate", "high")
    assert result.data["x"]["psi"] > 0


def test_categorical_drift_detected(tmp_path):
    """A large shift in categorical proportions must produce high/moderate severity."""
    rng: Generator = np.random.default_rng(42)
    reference = pl.DataFrame(
        {"cat": rng.choice(["A", "B"], size=1000, p=[0.8, 0.2]).tolist()},
    )
    current = pl.DataFrame(
        {"cat": rng.choice(["A", "B"], size=1000, p=[0.3, 0.7]).tolist()},
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectFeatureDrift,
        current_df=current,
        reference_df=reference,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "cat" in result.data
    assert result.data["cat"]["type"] == "categorical"
    assert result.data["cat"]["severity"] in ("moderate", "high")
    assert result.data["cat"]["tvd"] > 0.2


def test_skips_if_no_reference_data(tmp_path):
    """Task must return skipped status when no reference dataset is available."""
    current = pl.DataFrame({"x": [1, 2, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectFeatureDrift,
        current_df=current,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "skipped"
    assert "reference" in result.summary.get("message", "").lower()


def test_guidance_attached_for_high_drift_columns(tmp_path):
    """EDA guidance blurbs must be attached for columns with high drift severity."""
    rng: Generator = np.random.default_rng(42)
    reference = pl.DataFrame({"feature": rng.normal(0, 1, 1000).tolist()})
    current = pl.DataFrame({"feature": rng.normal(5, 1, 1000).tolist()})

    ctx, task = make_ctx_and_task(
        task_cls=DetectFeatureDrift,
        current_df=current,
        reference_df=reference,
        task_overrides={"psi": 0.1},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["feature"]["severity"] == "high"
    assert result.guidance is not None
    assert "feature" in result.guidance
    assert len(result.guidance["feature"]["eda"]) > 0
    assert result.guidance["feature"]["eda"][0]["level"] == "warn"


def test_no_drift_on_identical_distributions(tmp_path):
    """Identical current and reference distributions must produce low severity."""
    rng: Generator = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000).tolist()
    reference = pl.DataFrame({"x": data})
    current = pl.DataFrame({"x": data})

    ctx, task = make_ctx_and_task(
        task_cls=DetectFeatureDrift,
        current_df=current,
        reference_df=reference,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["x"]["severity"] == "low"
    assert result.summary["high_drift_columns"] == []


def test_no_shared_columns_returns_skipped(tmp_path):
    """Task must return skipped when current and reference share no columns."""
    current = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    reference = pl.DataFrame({"b": [4.0, 5.0, 6.0]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectFeatureDrift,
        current_df=current,
        reference_df=reference,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "skipped"


def test_no_plots_generated(tmp_path):
    """Feature drift task must not generate plots."""
    rng: Generator = np.random.default_rng(42)
    reference = pl.DataFrame({"x": rng.normal(0, 1, 100).tolist()})
    current = pl.DataFrame({"x": rng.normal(3, 1, 100).tolist()})

    ctx, task = make_ctx_and_task(
        task_cls=DetectFeatureDrift,
        current_df=current,
        reference_df=reference,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
