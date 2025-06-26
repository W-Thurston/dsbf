# tests/test_tasks/test_detect_high_cardinality.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_high_cardinality import DetectHighCardinality


def test_detect_high_cardinality_expected_output():
    """
    Test detection of high-cardinality columns using threshold=50.
    """
    df = pd.DataFrame(
        {
            "a": list(range(100)),  # 100 unique values
            "b": list("abcde") * 20,  # 5 unique values
        }
    )

    task = DetectHighCardinality(threshold=50)
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "a" in result.data
    assert "b" not in result.data
    assert result.metadata["threshold"] == 50
