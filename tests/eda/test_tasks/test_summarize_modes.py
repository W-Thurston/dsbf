# tests/test_tasks/test_summarize_modes.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_modes import SummarizeModes
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_modes_expected_output():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "y", "x", "z"]})

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    # Accept either scalar or multimodal list
    a_mode = result.data["a"]
    assert a_mode == 1 or (isinstance(a_mode, list) and 1 in a_mode)

    b_mode = result.data["b"]
    assert b_mode == "x" or (isinstance(b_mode, list) and "x" in b_mode)


def test_summarize_modes_multimodal_case():
    df = pd.DataFrame({"x": [1, 1, 2, 2, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert isinstance(result.data["x"], list)
    assert set(result.data["x"]) >= {1, 2}
