# tests/test_tasks/test_summarize_nulls.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_nulls import SummarizeNulls


def test_summarize_nulls_expanded_output():
    df = pd.DataFrame(
        {"a": [None, 1, None, 1], "b": [1, None, None, 1], "c": [1, 1, 1, 1]}
    )

    task = SummarizeNulls()
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "null_patterns" in result.data
    assert isinstance(result.data["null_patterns"], dict)
    assert "high_null_columns" in result.data
    assert any(col in ["a", "b"] for col in result.data["high_null_columns"])
