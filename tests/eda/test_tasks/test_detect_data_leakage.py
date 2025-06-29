# tests/test_tasks/test_detect_data_leakage.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_data_leakage import DetectDataLeakage
from tests.helpers.context_utils import make_ctx_and_task


def test_detect_data_leakage_expected_output():
    """
    Ensure DetectDataLeakage finds strongly correlated feature pairs.
    """
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],  # Perfectly correlated with x
            "z": [5, 4, 3, 2, 1],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectDataLeakage,
        current_df=df,
        task_overrides={"correlation_threshold": 0.99},
    )
    result = ctx.run_task(task)

    assert result is not None
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    leakage = result.data.get("leakage_pairs", {})
    assert isinstance(leakage, dict)
    assert any("x|y" in key or "y|x" in key for key in leakage)
