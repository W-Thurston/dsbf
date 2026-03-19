# tests/eda/test_tasks/test_detect_collinear_features.py

import warnings

import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_collinear_features import DetectCollinearFeatures
from tests.helpers.context_utils import make_ctx_and_task


@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in scalar divide:RuntimeWarning"
)
def test_detect_collinear_features_expected_output(tmp_path):
    """
    Test that DetectCollinearFeatures returns expected VIF flags
    for perfectly collinear variables.
    """
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],  # Strongly collinear with x1
            "x3": [5, 4, 3, 2, 1],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectCollinearFeatures,
        current_df=df,
        task_overrides={"vif_threshold": 5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result is not None, "No TaskResult returned"
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    scores = result.data.get("vif_scores", {})
    flagged = result.data.get("collinear_columns", [])

    assert isinstance(scores, dict)
    assert any(v > 5 for v in scores.values())
    assert "x2" in flagged or "x1" in flagged


def test_detect_collinear_features_core_output(tmp_path):
    """
    Confirm VIF scores and collinear column flags are produced correctly.
    """
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # Perfectly collinear with a
            "c": [5, 4, 3, 2, 1],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectCollinearFeatures,
        current_df=df,
        task_overrides={"vif_threshold": 5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered.*")
        result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None

    scores = result.data.get("vif_scores", {})
    flagged = result.data.get("collinear_columns", [])

    assert isinstance(scores, dict)
    assert len(scores) == 3  # all three columns scored
    assert any(v > 5 for v in scores.values())
    assert len(flagged) > 0

    # Guidance should be attached for high-VIF columns
    assert result.guidance is not None
    assert any(col in result.guidance for col in flagged)
