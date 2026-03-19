# tests/eda/test_tasks/test_detect_zeros.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_zeros import DetectZeros
from tests.helpers.context_utils import make_ctx_and_task


def test_zero_counts_and_flags_correct(tmp_path) -> None:
    """Zero counts, percentages, and flags must be computed correctly."""
    df = pd.DataFrame(
        {
            "a": [0, 0, 1, 2, 3, 0, 4, 0, 5, 0],  # 5 zeros → 50%
            "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 0 zeros
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectZeros,
        current_df=df,
        task_overrides={"flag_threshold": 0.3},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert result.data["zero_counts"]["a"] == 5
    assert result.data["zero_flags"]["a"] is True
    assert abs(result.data["zero_percentages"]["a"] - 0.5) < 1e-6
    assert result.data["zero_flags"]["b"] is False


def test_all_zeros_column_is_flagged(tmp_path) -> None:
    """A column where every value is zero must be flagged."""
    df = pd.DataFrame({"a": [0, 0, 0, 0], "b": [1, 2, 3, 4]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectZeros,
        current_df=df,
        task_overrides={"flag_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["zero_flags"]["a"] is True
    assert result.data["zero_flags"]["b"] is False


def test_guidance_attached_for_high_zero_columns(tmp_path) -> None:
    """EDA and ML guidance blurbs must be attached when zero rate ≥ 30%."""
    df = pd.DataFrame({"col": [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]})  # 30% zeros

    ctx, task = make_ctx_and_task(
        task_cls=DetectZeros,
        current_df=df,
        task_overrides={"flag_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    assert "col" in result.guidance
    assert len(result.guidance["col"]["eda"]) > 0
    assert len(result.guidance["col"]["ml"]) > 0
    ml_actions = result.guidance["col"]["ml"][0]["actions"]
    assert any(a["action"] == "transform" for a in ml_actions)


def test_no_guidance_for_low_zero_column(tmp_path) -> None:
    """Columns with zero rate below the guidance threshold must not emit blurbs."""
    df = pd.DataFrame({"col": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})  # 10% zeros

    ctx, task = make_ctx_and_task(
        task_cls=DetectZeros,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is None or "col" not in (result.guidance or {})


def test_no_plots_generated(tmp_path) -> None:
    """Zero detection task must not generate plots."""
    df = pd.DataFrame({"a": [0, 0, 1, 2, 3, 0, 4, 0, 5, 0]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectZeros,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None


def test_summary_message_has_correct_count(tmp_path) -> None:
    """Summary message must correctly count flagged columns."""
    df = pd.DataFrame(
        {
            "all_zeros": [0] * 10,  # 100% → flagged
            "mixed": [0] * 5 + [1] * 5,  # 50% → flagged at threshold 0.3
            "clean": list(range(10)),  # 10% → not flagged
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectZeros,
        current_df=df,
        task_overrides={"flag_threshold": 0.3},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "2" in result.summary["message"]
