# tests/test_tasks/test_detect_data_leakage.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_data_leakage import detect_data_leakage


def test_detect_data_leakage_expected_output():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],  # perfectly correlated with x
            "z": [5, 4, 3, 2, 1],
        }
    )

    result = detect_data_leakage(df, correlation_threshold=0.99)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    leakage = result.data["leakage_pairs"]
    assert any("x|y" in key or "y|x" in key for key in leakage)
