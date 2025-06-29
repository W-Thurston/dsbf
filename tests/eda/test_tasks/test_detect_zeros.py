# tests/test_tasks/test_detect_zeros.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_zeros import DetectZeros
from tests.helpers.context_utils import make_ctx_and_task


def test_detect_zeros_expected_output():
    df = pd.DataFrame(
        {
            "a": [0, 0, 1, 2, 3, 0, 4, 0, 5, 0],  # 5 zeros -> 50%
            "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 0 zeros
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectZeros, current_df=df, task_overrides={"flag_threshold": 0.3}
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    counts = result.data["zero_counts"]
    flags = result.data["zero_flags"]
    percentages = result.data["zero_percentages"]

    assert counts["a"] == 5
    assert flags["a"] is True
    assert percentages["a"] == 0.5
    assert flags["b"] is False


def test_detect_zeros_all_zeros_or_none():
    df = pd.DataFrame({"a": [0, 0, 0, 0], "b": [1, 2, 3, 4]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectZeros, current_df=df, task_overrides={"flag_threshold": 0.5}
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.data["zero_flags"]["a"] is True
    assert result.data["zero_flags"]["b"] is False
