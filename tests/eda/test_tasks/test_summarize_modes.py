# tests/test_tasks/test_summarize_modes.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_modes import summarize_modes


def test_summarize_modes_expected_output():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "y", "x", "z"]})

    result = summarize_modes(df)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["a"] == 1 or (
        isinstance(result.data["a"], list) and 1 in result.data["a"]
    )
    assert result.data["b"] == "x" or (
        isinstance(result.data["b"], list) and "x" in result.data["b"]
    )
