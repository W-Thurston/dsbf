# tests/test_tasks/test_detect_duplicates.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_duplicates import DetectDuplicates


def test_detect_duplicates_expected_output():
    """
    Ensure that duplicate rows are correctly counted.
    """
    df = pd.DataFrame(
        {
            "a": [1, 1, 2, 3, 3],
            "b": ["x", "x", "y", "z", "z"],  # (1, x) and (3, z) are duplicates
        }
    )

    task = DetectDuplicates()
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert result is not None
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["duplicate_count"] == 2
