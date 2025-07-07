# tests/eda/test_tasks/test_detect_feature_drift.py

from pathlib import Path

import numpy as np
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_feature_drift import DetectFeatureDrift
from tests.helpers.context_utils import make_ctx_and_task


def test_numeric_drift_detected():
    np.random.seed(0)
    reference = pl.DataFrame({"x": np.random.normal(0, 1, 1000)})
    current = pl.DataFrame({"x": np.random.normal(3, 1, 1000)})

    ctx, task = make_ctx_and_task(
        task_cls=DetectFeatureDrift, current_df=current, reference_df=reference
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None, "Expected result.data to be present"
    assert "x" in result.data
    assert result.data["x"]["type"] == "numerical"
    assert result.data["x"]["severity"] in {"moderate", "high"}
    assert result.data["x"]["psi"] > 0


def test_categorical_drift_detected():
    np.random.seed(0)
    reference = pl.DataFrame(
        {"cat": np.random.choice(["A", "B"], size=1000, p=[0.8, 0.2])}
    )
    current = pl.DataFrame(
        {"cat": np.random.choice(["A", "B"], size=1000, p=[0.3, 0.7])}
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectFeatureDrift, current_df=current, reference_df=reference
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None, "Expected result.data to be present"
    assert "cat" in result.data
    assert result.data["cat"]["type"] == "categorical"
    assert result.data["cat"]["severity"] in {"moderate", "high"}
    assert result.data["cat"]["tvd"] > 0.2


def test_skips_if_no_reference_data():
    current = pl.DataFrame({"x": [1, 2, 3]})
    ctx, task = make_ctx_and_task(
        task_cls=DetectFeatureDrift,
        current_df=current,
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result is not None
    assert result.status == "skipped"
    assert "reference" in result.summary.get("message", "").lower()


def test_numeric_drift_generates_plot(tmp_path):
    """
    Confirm that numeric drift triggers plot generation for shared columns.
    """
    np.random.seed(42)
    reference = pl.DataFrame({"feature": np.random.normal(0, 1, 1000)})
    current = pl.DataFrame({"feature": np.random.normal(2, 1, 1000)})

    ctx, task = make_ctx_and_task(
        task_cls=DetectFeatureDrift,
        current_df=current,
        reference_df=reference,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "feature" in result.plots

    plot_entry = result.plots["feature"]

    # Static plot
    static_path = plot_entry["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
    assert static_path.suffix == ".png"

    # Interactive structure
    interactive = plot_entry["interactive"]
    assert isinstance(interactive, dict)
    assert interactive["type"] == "histogram"
    assert "data" in interactive
    assert "config" in interactive
    assert "annotations" in interactive
    assert any("PSI" in ann for ann in interactive["annotations"])
