# tests/test_tasks/test_summarize_modes.py

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_modes import SummarizeModes


def test_summarize_modes_expected_output():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "y", "x", "z"]})

    context = AnalysisContext(df)
    result = context.run_task(SummarizeModes())

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    # Accept either scalar or multimodal list
    a_mode = result.data["a"]
    assert a_mode == 1 or (isinstance(a_mode, list) and 1 in a_mode)

    b_mode = result.data["b"]
    assert b_mode == "x" or (isinstance(b_mode, list) and "x" in b_mode)
