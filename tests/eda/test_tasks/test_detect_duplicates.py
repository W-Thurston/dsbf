# tests/test_tasks/test_detect_duplicates.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_duplicates import detect_duplicates


def test_detect_duplicates_expected_output():
    df = pd.DataFrame({"a": [1, 1, 2, 3, 3], "b": ["x", "x", "y", "z", "z"]})

    result = detect_duplicates(df)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["duplicate_count"] == 2
