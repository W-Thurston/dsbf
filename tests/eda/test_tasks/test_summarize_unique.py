# tests/test_tasks/test_summarize_unique.py

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_unique import SummarizeUnique


def test_summarize_unique_expected_output():
    df = pd.DataFrame(
        {"a": [1, 2, 2, 3], "b": ["x", "x", "y", "z"], "c": [True, False, True, True]}
    )

    context = AnalysisContext(df)
    result = context.run_task(SummarizeUnique())

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["a"] == 3
    assert result.data["b"] == 3
    assert result.data["c"] == 2


def test_summarize_unique_empty_column():
    df = pd.DataFrame({"empty": [None, None, None, None]})
    context = AnalysisContext(df)
    result = context.run_task(SummarizeUnique())
    assert result.status == "success"
    assert result.data["empty"] == 0
