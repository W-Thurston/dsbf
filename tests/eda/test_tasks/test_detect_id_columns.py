# tests/test_tasks/test_detect_id_columns.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_id_columns import detect_id_columns


def test_detect_id_columns_expected_output():
    df = pd.DataFrame(
        {
            "id": range(100),  # likely ID
            "uuid": [f"id_{i}" for i in range(100)],  # also likely ID
            "group": [1] * 100,  # not ID
        }
    )

    result = detect_id_columns(df)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "id" in result.data
    assert "uuid" in result.data
    assert "group" not in result.data
