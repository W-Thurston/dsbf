# tests/test_tasks/test_detect_duplicate_columns.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_duplicate_columns import detect_duplicate_columns


def test_detect_duplicate_columns_expected_output():
    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": [1, 2, 3], "c": [3, 2, 1]}  # duplicate of a
    )

    result = detect_duplicate_columns(df)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    pairs = result.data["duplicate_column_pairs"]
    assert ("a", "b") in pairs or ("b", "a") in pairs
