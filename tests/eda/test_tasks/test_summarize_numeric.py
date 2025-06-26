# tests/test_tasks/test_summarize_numeric.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_numeric import SummarizeNumeric


def test_summarize_numeric_expected_output():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 10, 10, 10, 10],  # zero variance
            "c": [100, 200, 300, 400, 500],
        }
    )

    task = SummarizeNumeric()
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "a" in result.data
    assert "1%" in result.data["a"]
    assert "near_zero_variance" in result.data["b"]
    assert result.data["b"]["near_zero_variance"] is True
