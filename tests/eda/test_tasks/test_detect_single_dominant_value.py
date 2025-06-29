# tests/test_tasks/test_detect_single_dominant_value.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_single_dominant_value import DetectSingleDominantValue
from tests.helpers.context_utils import make_ctx_and_task


def test_detect_single_dominant_value_expected_output():
    df = pd.DataFrame(
        {
            "mostly_ones": [1] * 95 + [0] * 5,
            "uniform": [1, 2, 3, 4, 5] * 20,
            "binary": ["yes"] * 96 + ["no"] * 4,
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectSingleDominantValue,
        current_df=df,
        task_overrides={"dominance_threshold": 0.9},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "mostly_ones" in result.data
    assert "binary" in result.data
    assert "uniform" not in result.data
