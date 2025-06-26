# tests/test_tasks/test_detect_zeros.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_zeros import detect_zeros


def test_detect_zeros_expected_output():
    df = pd.DataFrame(
        {
            "a": [0, 0, 1, 2, 3, 0, 4, 0, 5, 0],  # 50% zeros
            "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 0% zeros
        }
    )

    result = detect_zeros(df, flag_threshold=0.3)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    counts = result.data["zero_counts"]
    flags = result.data["zero_flags"]
    percentages = result.data["zero_percentages"]

    assert counts["a"] == 5
    assert flags["a"] is True
    assert percentages["a"] == 0.5
    assert flags["b"] is False
