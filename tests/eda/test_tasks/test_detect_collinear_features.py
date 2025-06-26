# tests/test_tasks/test_detect_collinear_features.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_collinear_features import detect_collinear_features


def test_detect_collinear_features_expected_output():
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],  # Perfectly collinear with x1
            "x3": [5, 4, 3, 2, 1],
        }
    )

    result = detect_collinear_features(df, vif_threshold=5)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "vif_scores" in result.data
    assert any(vif > 5 for vif in result.data["vif_scores"].values())
    assert (
        "x2" in result.data["collinear_columns"]
        or "x1" in result.data["collinear_columns"]
    )
