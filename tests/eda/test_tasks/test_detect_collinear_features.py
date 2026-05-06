# tests/eda/test_tasks/test_detect_collinear_features.py

import warnings
from collections.abc import Generator
from typing import Any

import pandas as pd
import pytest
from numpy import dtype, ndarray

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_collinear_features import DetectCollinearFeatures
from tests.helpers.context_utils import make_ctx_and_task


@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in scalar divide:RuntimeWarning"
)
def test_detect_collinear_features_expected_output(tmp_path) -> None:
    """
    Test that DetectCollinearFeatures returns expected VIF flags
    for strongly collinear variables.

    Uses 30 rows of floats so infer_types classifies columns as
    continuous rather than ID-like (unique_ratio would be 1.0 on
    small integer sequences, causing get_columns_by_intent to return
    zero eligible columns).
    """
    import numpy as np

    rng: Generator = np.random.default_rng(42)
    base: ndarray[tuple[Any, ...], dtype[float]] = rng.normal(0, 1, 30)
    df = pd.DataFrame(
        {
            "x1": base,
            "x2": base * 2 + rng.normal(0, 0.01, 30),  # near-collinear
            "x3": rng.normal(0, 1, 30),  # independent
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectCollinearFeatures,
        current_df=df,
        task_overrides={"vif_threshold": 5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result is not None, "No TaskResult returned"
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    scores = result.data.get("vif_scores", {})
    flagged = result.data.get("collinear_columns", [])

    assert isinstance(scores, dict)
    assert any(v > 5 for v in scores.values())
    assert "x2" in flagged or "x1" in flagged


def test_detect_collinear_features_core_output(tmp_path) -> None:
    """
    Confirm VIF scores and collinear column flags are produced correctly.

    Uses 30 rows of floats so infer_types classifies columns as continuous.
    """
    import numpy as np

    rng: Generator = np.random.default_rng(0)
    base: ndarray[tuple[Any, ...], dtype[float]] = rng.normal(0, 1, 30)
    df = pd.DataFrame(
        {
            "a": base,
            "b": base * 2 + rng.normal(0, 0.01, 30),  # near-perfectly collinear
            "c": rng.normal(0, 1, 30),  # independent
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
