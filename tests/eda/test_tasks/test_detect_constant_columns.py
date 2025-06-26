# tests/test_tasks/test_detect_constant_columns.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_constant_columns import detect_constant_columns


def test_detect_constant_columns_expected_output():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": ["x", "x", "x"]})

    result = detect_constant_columns(df)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert sorted(result.data["constant_columns"]) == ["a", "c"]
