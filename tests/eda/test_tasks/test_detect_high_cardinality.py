# tests/test_tasks/test_detect_high_cardinality.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_high_cardinality import detect_high_cardinality


def test_detect_high_cardinality_expected_output():
    df = pd.DataFrame(
        {
            "a": list(range(100)),  # high cardinality
            "b": list("abcde") * 20,  # low cardinality
        }
    )

    result = detect_high_cardinality(df, threshold=50)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "a" in result.data
    assert "b" not in result.data
    assert result.metadata["threshold"] == 50
