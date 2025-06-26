# tests/test_tasks/test_detect_id_columns.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_id_columns import DetectIDColumns


def test_detect_id_columns_expected_output():
    df = pd.DataFrame(
        {
            "id": range(100),  # unique IDs
            "uuid": [f"id_{i}" for i in range(100)],  # also unique
            "group": [1] * 100,  # constant (not ID)
        }
    )

    task = DetectIDColumns(threshold_ratio=0.95)
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "id" in result.data
    assert "uuid" in result.data
    assert "group" not in result.data
    assert result.metadata["threshold_ratio"] == 0.95
