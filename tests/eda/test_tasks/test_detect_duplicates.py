# tests/test_tasks/test_detect_duplicates.py

import pandas as pd

from dsbf.core.context import AnalysisContext
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

    context = AnalysisContext(df)
    result = context.run_task(DetectDuplicates())

    assert result is not None
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["duplicate_count"] == 2


def test_detect_duplicates_no_duplicates():
    """Ensure that a DataFrame with no duplicate rows returns 0."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        }
    )
    context = AnalysisContext(df)
    result = context.run_task(DetectDuplicates())

    assert result.status == "success"
    assert result.data["duplicate_count"] == 0
