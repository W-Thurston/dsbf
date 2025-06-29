# tests/test_tasks/test_detect_high_cardinality.py

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_high_cardinality import DetectHighCardinality


def test_detect_high_cardinality_expected_output():
    """
    Test detection of high-cardinality columns using cardinality_threshold=50.
    """
    df = pd.DataFrame(
        {
            "a": list(range(100)),  # 100 unique values
            "b": list("abcde") * 20,  # 5 unique values
        }
    )

    context = AnalysisContext(df)
    result = context.run_task(
        DetectHighCardinality(config={"cardinality_threshold": 50})
    )

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "a" in result.data
    assert "b" not in result.data
    assert result.metadata["cardinality_threshold"] == 50


def test_detect_high_cardinality_with_low_cardinality_column():
    df = pd.DataFrame({"c": ["x", "x", "y", "z", "x", "y", "z"]})
    context = AnalysisContext(df)
    result = context.run_task(
        DetectHighCardinality(config={"cardinality_threshold": 10})
    )
    assert result.status == "success"
    assert "c" not in result.data
