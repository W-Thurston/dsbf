# tests/eda/test_tasks/test_detect_single_dominant_value.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_single_dominant_value import DetectSingleDominantValue
from tests.helpers.context_utils import make_ctx_and_task


def test_dominant_columns_flagged_in_summary(tmp_path) -> None:
    """Columns exceeding the dominance threshold must be counted in summary message."""
    df = pd.DataFrame(
        {
            "mostly_ones": [1] * 95 + [0] * 5,
            "uniform": [1, 2, 3, 4, 5] * 20,
            "binary": ["yes"] * 96 + ["no"] * 4,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectSingleDominantValue,
        current_df=df,
        task_overrides={"dominance_threshold": 0.9},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    # All columns are stored in data — check proportions rather than presence/absence
    assert result.data["mostly_ones"]["mode_proportion"] >= 0.9
    assert result.data["binary"]["mode_proportion"] >= 0.9
    # uniform has 5 equal values — proportion ~0.2, well below threshold
    assert result.data["uniform"]["mode_proportion"] < 0.9

    # Summary count should reflect dominant columns
    assert "2 column(s)" in result.summary["message"]


def test_all_null_column_produces_empty_data(tmp_path) -> None:
    """An all-null column must be skipped and return empty data."""
    df = pd.DataFrame({"col": [None, None, None]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectSingleDominantValue,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data == {}
    assert result.plots is None


def test_constant_column_stored_with_full_dominance(tmp_path) -> None:
    """A constant column must be stored with mode_proportion of 1.0."""
    df = pd.DataFrame({"col": ["A"] * 100})

    ctx, task = make_ctx_and_task(
        task_cls=DetectSingleDominantValue,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert "col" in result.data
    assert result.data["col"]["mode_proportion"] == 1.0


def test_guidance_attached_for_highly_dominant_columns(tmp_path) -> None:
    """EDA and ML guidance blurbs must be attached when mode proportion ≥ 0.7."""
    df = pd.DataFrame({"col": ["A"] * 90 + ["B"] * 10})

    ctx, task = make_ctx_and_task(
        task_cls=DetectSingleDominantValue,
        current_df=df,
        task_overrides={"dominance_threshold": 0.95},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    assert "col" in result.guidance
    assert len(result.guidance["col"]["eda"]) > 0
    assert len(result.guidance["col"]["ml"]) > 0


def test_mode_and_proportion_recorded_correctly(tmp_path) -> None:
    """Mode and mode_proportion must accurately reflect the most common value."""
    df = pd.DataFrame({"votes": ["yes"] * 80 + ["no"] * 20})

    ctx, task = make_ctx_and_task(
        task_cls=DetectSingleDominantValue,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["votes"]["mode"] == "yes"
    assert abs(result.data["votes"]["mode_proportion"] - 0.8) < 0.01


def test_no_plots_generated(tmp_path) -> None:
    """Single dominant value task must not generate plots."""
    df = pd.DataFrame({"col": ["A"] * 90 + ["B"] * 10})

    ctx, task = make_ctx_and_task(
        task_cls=DetectSingleDominantValue,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
