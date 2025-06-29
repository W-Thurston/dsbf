# tests/eda/test_tasks/test_detect_feature_drift.py

import numpy as np
import polars as pl

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_feature_drift import DetectFeatureDrift


def create_test_context(current_df, reference_df, task_cfg={}):
    ctx = AnalysisContext(
        data=current_df,
        config={"tasks": {"detect_feature_drift": task_cfg}},
        output_dir="./dsbf/outputs/test",
        run_metadata={},
    )
    ctx.reference_data = reference_df
    return ctx


def run_drift_task(current_df, reference_df, task_cfg={}) -> TaskResult:
    ctx = create_test_context(current_df, reference_df, task_cfg)
    task = DetectFeatureDrift(name="detect_feature_drift", config=task_cfg)
    task.set_input(current_df)
    task.context = ctx
    task.run()
    result = task.get_output()
    if result is None:
        raise RuntimeError("Task did not produce a result.")

    return result


def test_numeric_drift_detected():
    np.random.seed(0)
    reference = pl.DataFrame({"x": np.random.normal(0, 1, 1000)})
    current = pl.DataFrame({"x": np.random.normal(3, 1, 1000)})

    result = run_drift_task(current, reference)

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

    result = run_drift_task(current, reference)

    assert result.status == "success"
    assert result.data is not None, "Expected result.data to be present"
    assert "cat" in result.data
    assert result.data["cat"]["type"] == "categorical"
    assert result.data["cat"]["severity"] in {"moderate", "high"}
    assert result.data["cat"]["tvd"] > 0.2


def test_skips_if_no_reference_data():
    current = pl.DataFrame({"x": [1, 2, 3]})
    ctx = AnalysisContext(data=current, config={}, output_dir=".", run_metadata={})

    task = DetectFeatureDrift(name="detect_feature_drift")
    task.set_input(current)
    task.context = ctx
    task.run()
    result = task.get_output()

    assert result is not None
    assert result.status == "skipped"
    assert "reference" in result.summary.get("message", "").lower()
