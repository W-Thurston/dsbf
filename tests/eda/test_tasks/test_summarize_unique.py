# tests/test_tasks/test_summarize_unique.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_unique import SummarizeUnique


def test_summarize_unique_expected_output():
    df = pd.DataFrame(
        {"a": [1, 2, 2, 3], "b": ["x", "x", "y", "z"], "c": [True, False, True, True]}
    )

    task = SummarizeUnique()
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["a"] == 3
    assert result.data["b"] == 3
    assert result.data["c"] == 2
