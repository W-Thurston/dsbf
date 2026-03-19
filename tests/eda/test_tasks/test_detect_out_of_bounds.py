# tests/eda/test_tasks/test_detect_out_of_bounds.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_out_of_bounds import DetectOutOfBounds
from tests.helpers.context_utils import make_ctx_and_task


def test_violations_detected_for_named_columns(tmp_path) -> None:
    """Columns matching default rule names with out-of-range values must be flagged."""
    df = pd.DataFrame(
        {
            "age": [25, 30, -5, 150],  # -5 and 150 are violations
            "score": [0.8, 0.95, 1.1, 0.5],  # 1.1 is a violation
            "percent": [50, 110, 20, -10],  # 110 and -10 are violations
            "weight": [150, 180, 200, 175],  # no default rule — should be ignored
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutOfBounds,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "age" in result.data
    assert result.data["age"]["count"] == 2
    assert "score" in result.data
    assert "percent" in result.data
    assert "weight" not in result.data  # no default rule for weight


def test_no_violations_on_clean_data(tmp_path) -> None:
    """Columns with all values within bounds must produce no flagged results."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 45, 65],
            "percent": [10, 20, 50, 99],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutOfBounds,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data == {}


def test_custom_bounds_respected(tmp_path) -> None:
    """Custom bounds from task config must override defaults."""
    df = pd.DataFrame({"height": [160, 175, 250, 300]})  # 250 and 300 violate (0, 220)

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutOfBounds,
        current_df=df,
        task_overrides={"custom_bounds": {"height": (0, 220)}},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "height" in result.data
    assert result.data["height"]["count"] == 2


def test_guidance_attached_for_violations(tmp_path) -> None:
    """Both EDA and ML guidance blurbs must be attached for violated columns."""
    df = pd.DataFrame({"age": [25, -5, 30, 200]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutOfBounds,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "age" in result.data
    assert result.guidance is not None
    assert "age" in result.guidance
    assert len(result.guidance["age"]["eda"]) > 0
    assert len(result.guidance["age"]["ml"]) > 0
    ml_actions = result.guidance["age"]["ml"][0]["actions"]
    action_types: set = {a["action"] for a in ml_actions}
    assert "winsorize" in action_types or "investigate" in action_types


def test_no_plots_generated(tmp_path) -> None:
    """Out-of-bounds detection must not generate plots."""
    df = pd.DataFrame({"age": [25, -5, 30, 200]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectOutOfBounds,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
