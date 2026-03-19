# tests/eda/test_tasks/test_detect_outliers.py

import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_outliers import DetectOutliers
from tests.helpers.context_utils import make_ctx_and_task


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_detect_outliers_expected_output(tmp_path) -> None:
    """
    Columns with extreme values must be flagged.

    Tightly clustered columns must not.
    """
    df = pd.DataFrame(
        {
            "normal": [10, 12, 11, 13, 12, 11, 10],
            "outlier_col": [100, 101, 102, 103, 1000, 104, 105],  # 1000 is an outlier
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

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


def test_detect_outliers_no_outliers(tmp_path) -> None:
    """A tightly clustered column must not be flagged."""
    df = pd.DataFrame({"stable": [10, 11, 10, 11, 10, 11, 10]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["outlier_flags"]["stable"] is False


def test_detect_outliers_empty_dataframe(tmp_path):
    """An empty DataFrame must return success with all-empty result dicts."""
    df = pd.DataFrame()

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["outlier_counts"] == {}
    assert result.data["outlier_flags"] == {}
    assert result.data["outlier_rows"] == {}
    assert result.plots is None


def test_detect_outliers_all_null_columns(tmp_path) -> None:
    """Columns that are entirely null must be skipped without error."""
    df = pd.DataFrame({"a": [None, None, None], "b": [float("nan")] * 3})

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    # Empty-after-dropna columns are skipped entirely
    assert result.data["outlier_counts"].get("b", 0) == 0
    assert result.plots is None


def test_detect_outliers_non_numeric_columns(tmp_path) -> None:
    """Non-numeric columns must produce empty outlier dicts."""
    df = pd.DataFrame({"name": ["alice", "bob", "carol"], "category": ["x", "y", "z"]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["outlier_counts"] == {}
    assert result.data["outlier_flags"] == {}
    assert result.data["outlier_rows"] == {}
    assert result.plots is None


def test_guidance_attached_for_outlier_columns(tmp_path) -> None:
    """EDA and ML guidance blurbs must be attached for columns with outliers."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 100]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["outlier_counts"]["x"] >= 1
    assert result.guidance is not None
    assert "x" in result.guidance
    assert len(result.guidance["x"]["eda"]) > 0
    assert len(result.guidance["x"]["ml"]) > 0
    ml_actions = result.guidance["x"]["ml"][0]["actions"]
    assert any(a["action"] == "winsorize" for a in ml_actions)


def test_no_plots_generated(tmp_path) -> None:
    """Outlier detection task must not generate plots."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 100], "y": [10, 12, 14, 13, 15]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
