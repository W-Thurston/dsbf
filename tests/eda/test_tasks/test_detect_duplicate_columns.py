# tests/test_tasks/test_detect_duplicate_columns.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_duplicate_columns import DetectDuplicateColumns


def test_detect_duplicate_columns_expected_output():
    """
    Test that DetectDuplicateColumns identifies duplicate column pairs.
    """
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1, 2, 3],  # Duplicate of 'a'
            "c": [3, 2, 1],
        }
    )

    task = DetectDuplicateColumns()
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert result is not None
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    pairs = result.data.get("duplicate_column_pairs", [])
    assert any(set(pair) == {"a", "b"} for pair in pairs)
