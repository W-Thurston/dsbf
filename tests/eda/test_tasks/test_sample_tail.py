# tests/test_tasks/test_sample_tail.py

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.sample_tail import SampleTail


def test_sample_tail_expected_output():
    df = pd.DataFrame({"a": list(range(10))})

    context = AnalysisContext(df)
    result = context.run_task(SampleTail(config={"n": 3}))

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert len(result.data["sample"]["a"]) == 3
    assert result.data["sample"]["a"] == [7, 8, 9]


def test_sample_tail_zero():
    df = pd.DataFrame({"a": [1, 2, 3]})
    context = AnalysisContext(df)

    result_zero = context.run_task(SampleTail(config={"n": 0}))
    assert result_zero.status == "success"
    assert result_zero.data is not None
    assert len(result_zero.data["sample"]["a"]) == 0


def test_sample_tail_oversample():
    df = pd.DataFrame({"a": [1, 2, 3]})
    context = AnalysisContext(df)

    result_oversample = context.run_task(SampleTail(config={"n": 10}))
    assert result_oversample.status == "success"
    assert result_oversample.data is not None
    assert result_oversample.data["sample"]["a"] == [1, 2, 3]
