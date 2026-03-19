# tests/eda/test_tasks/test_summarize_boolean_fields.py

from typing import TYPE_CHECKING

import pandas as pd

from dsbf.eda.tasks.summarize_boolean_fields import SummarizeBooleanFields
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_balanced_binary_column_summarized(tmp_path) -> None:
    """A balanced binary column must have pct_true and pct_false near 0.5."""
    df = pd.DataFrame({"flag": [True, False] * 50})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeBooleanFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeBooleanFields)

    assert result.status == "success"
    assert "flag" in result.data
    assert abs(result.data["flag"]["pct_true"] - 0.5) < 0.01
    assert abs(result.data["flag"]["pct_false"] - 0.5) < 0.01


def test_imbalanced_column_emits_guidance(tmp_path) -> None:
    """A column with ≥ 75% one class must emit EDA and ML guidance blurbs."""
    df = pd.DataFrame({"flag": [True] * 90 + [False] * 10})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeBooleanFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeBooleanFields)

    assert result.status == "success"
    assert result.guidance is not None
    assert "flag" in result.guidance
    assert len(result.guidance["flag"]["eda"]) > 0
    assert len(result.guidance["flag"]["ml"]) > 0
    ml_actions = result.guidance["flag"]["ml"][0]["actions"]
    assert any(a["action"] == "stratified_split" for a in ml_actions)


def test_balanced_column_produces_no_guidance(tmp_path) -> None:
    """A balanced binary column (< 75% one class) must not emit guidance."""
    df = pd.DataFrame({"x": [0, 1] * 50})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeBooleanFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeBooleanFields)

    assert result.status == "success"
    assert result.guidance is None or "x" not in (result.guidance or {})


def test_null_percentage_tracked(tmp_path) -> None:
    """pct_null must reflect the actual proportion of null values."""
    df = pd.DataFrame({"flag": [True, False, None, True, None]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeBooleanFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeBooleanFields)

    assert result.status == "success"
    assert abs(result.data["flag"]["pct_null"] - 0.4) < 0.01


def test_no_plots_generated(tmp_path) -> None:
    """Boolean summary must not generate plots."""
    df = pd.DataFrame({"flag": [True, False] * 10})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeBooleanFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeBooleanFields)

    assert result.status == "success"
    assert result.plots is None
