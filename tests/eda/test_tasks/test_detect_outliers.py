# tests/test_tasks/test_detect_outliers.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_outliers import detect_outliers


def test_detect_outliers_expected_output():
    df = pd.DataFrame(
        {
            "normal": [10, 12, 11, 13, 12, 11, 10],
            "outlier_col": [100, 101, 102, 103, 1000, 104, 105],  # 1000 is an outlier
        }
    )

    result = detect_outliers(df)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    counts = result.data["outlier_counts"]
    flags = result.data["outlier_flags"]
    rows = result.data["outlier_rows"]

    assert "outlier_col" in counts
    assert counts["outlier_col"] >= 1
    assert flags["outlier_col"] is True
    assert any(idx in rows["outlier_col"] for idx in range(len(df)))
