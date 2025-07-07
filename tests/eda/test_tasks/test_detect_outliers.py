# tests/eda/test_tasks/test_detect_outliers.py

import warnings
from pathlib import Path

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_outliers import DetectOutliers
from tests.helpers.context_utils import make_ctx_and_task


def test_detect_outliers_expected_output(tmp_path):
    warnings.filterwarnings(
        "ignore", category=PendingDeprecationWarning, module="seaborn"
    )
    df = pd.DataFrame(
        {
            "normal": [10, 12, 11, 13, 12, 11, 10],
            "outlier_col": [100, 101, 102, 103, 1000, 104, 105],  # 1000 is an outlier
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    counts = result.data["outlier_counts"]
    flags = result.data["outlier_flags"]
    rows = result.data["outlier_rows"]

    assert "outlier_col" in counts
    assert counts["outlier_col"] >= 1
    assert flags["outlier_col"] is True
    assert isinstance(rows["outlier_col"], list)
    assert any(idx in rows["outlier_col"] for idx in range(len(df)))


def test_detect_outliers_no_outliers(tmp_path):
    warnings.filterwarnings(
        "ignore", category=PendingDeprecationWarning, module="seaborn"
    )
    df = pd.DataFrame({"stable": [10, 11, 10, 11, 10, 11, 10]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["outlier_flags"]["stable"] is False


def test_detect_outliers_empty_df(tmp_path):
    df = pd.DataFrame()
    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["outlier_counts"] == {}
    assert result.data["outlier_flags"] == {}
    assert result.data["outlier_rows"] == {}
    assert result.plots == {}


def test_detect_outliers_all_nulls(tmp_path):
    df = pd.DataFrame({"a": [None, None, None], "b": [float("nan")] * 3})
    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["outlier_counts"].get("b", 0) == 0
    assert result.data["outlier_rows"].get("b", []) == []
    assert result.data["outlier_flags"] == {}
    assert result.data["outlier_rows"] == {}
    assert result.plots == {}


def test_detect_outliers_non_numeric(tmp_path):
    df = pd.DataFrame({"name": ["alice", "bob", "carol"], "category": ["x", "y", "z"]})
    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["outlier_counts"] == {}
    assert result.data["outlier_flags"] == {}
    assert result.data["outlier_rows"] == {}
    assert result.plots == {}


def test_detect_outliers_with_plots(tmp_path):
    warnings.filterwarnings(
        "ignore", category=PendingDeprecationWarning, module="seaborn"
    )
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 100],
            "y": [10, 12, 14, 13, 15],
            "z": [20, 25, 22, 24, 500],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert isinstance(result.data, dict)
    assert "x" in result.data["outlier_counts"]
    assert result.plots is not None
    assert "x" in result.plots

    # Check static file exists and remove it
    static_path = result.plots["x"]["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
    static_path.unlink()

    # Check annotations in interactive plot
    interactive = result.plots["x"]["interactive"]
    assert isinstance(interactive, dict)
    assert "annotations" in interactive
    assert any("Outliers detected" in a for a in interactive["annotations"])
