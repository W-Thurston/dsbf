# tests/test_tasks/test_detect_outliers.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_outliers import DetectOutliers
from tests.helpers.context_utils import make_ctx_and_task


def test_detect_outliers_expected_output():
    df = pd.DataFrame(
        {
            "normal": [10, 12, 11, 13, 12, 11, 10],
            "outlier_col": [100, 101, 102, 103, 1000, 104, 105],  # 1000 is an outlier
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
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


def test_detect_outliers_no_outliers():
    df = pd.DataFrame({"stable": [10, 11, 10, 11, 10, 11, 10]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["outlier_flags"]["stable"] is False
