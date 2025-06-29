# tests/test_tasks/test_detect_id_columns.py

import pandas as pd

from dsbf.core.context import AnalysisContext
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

    context = AnalysisContext(df)
    result = context.run_task(DetectIDColumns(config={"threshold_ratio": 0.95}))

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "id" in result.data
    assert "uuid" in result.data
    assert "group" not in result.data
    assert result.metadata["threshold_ratio"] == 0.95


def test_detect_id_columns_handles_no_ids():
    df = pd.DataFrame({"group": [1, 1, 1, 1], "value": [10, 10, 20, 20]})
    context = AnalysisContext(df)
    result = context.run_task(DetectIDColumns(config={"threshold_ratio": 0.95}))
    assert result.status == "success"
    assert result.data == {}
