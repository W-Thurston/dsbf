# tests/test_tasks/test_sample_head.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.sample_head import SampleHead
from tests.helpers.context_utils import make_ctx_and_task


def test_sample_head_expected_output():
    df = pd.DataFrame({"a": list(range(10))})

    ctx, task = make_ctx_and_task(
        task_cls=SampleHead, current_df=df, task_overrides={"n": 3}
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert len(result.data["sample"]["a"]) == 3
    assert result.data["sample"]["a"] == [0, 1, 2]


def test_sample_head_oversample_handled_gracefully():
    df = pd.DataFrame({"a": [1, 2]})

    ctx, task = make_ctx_and_task(
        task_cls=SampleHead, current_df=df, task_overrides={"n": 10}
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert len(result.data["sample"]["a"]) == 2
