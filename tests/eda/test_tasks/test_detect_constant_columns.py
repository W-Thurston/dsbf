# tests/test_tasks/test_detect_constant_columns.py

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_constant_columns import DetectConstantColumns


def test_detect_constant_columns_expected_output():
    """
    Test that DetectConstantColumns correctly identifies columns
    with only one unique value.
    """
    df = pd.DataFrame(
        {
            "a": [1, 1, 1],  # Constant
            "b": [1, 2, 3],  # Varying
            "c": ["x", "x", "x"],  # Constant
        }
    )

    context = AnalysisContext(df)
    result = context.run_task(DetectConstantColumns())

    assert result is not None, "No TaskResult returned"
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert sorted(result.data["constant_columns"]) == ["a", "c"]
