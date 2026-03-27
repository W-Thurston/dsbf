# tests/eda/test_tasks/test_detect_duplicate_columns.py

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_duplicate_columns import DetectDuplicateColumns
from tests.helpers.context_utils import make_ctx_and_task


def test_duplicate_pair_detected(tmp_path):
    """An exact duplicate column pair must appear in duplicate_column_pairs."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1, 2, 3],  # exact duplicate of a
            "c": [3, 2, 1],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    pairs = result.data["duplicate_column_pairs"]
    assert any(set(pair) == {"a", "b"} for pair in pairs)
    # c is distinct - must not appear in any pair
    assert not any("c" in pair for pair in pairs)


def test_no_duplicate_columns(tmp_path):
    """A dataset with no duplicate columns must return an empty list."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["duplicate_column_pairs"] == []


def test_guidance_attached_for_duplicate_pair(tmp_path):
    """EDA and ML guidance blurbs must be attached for duplicate columns."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None

    # At least one column in the pair must have guidance
    guided_cols: set[str] = set(result.guidance.keys())
    assert guided_cols & {"a", "b"}

    for col in guided_cols & {"a", "b"}:
        assert len(result.guidance[col]["eda"]) > 0
        assert len(result.guidance[col]["ml"]) > 0
        drop_actions = result.guidance[col]["ml"][0]["actions"]
        assert any(a["action"] == "drop" for a in drop_actions)


def test_null_values_handled_correctly(tmp_path):
    """Columns with nulls in the same positions must still be detected as duplicates."""
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, None, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    pairs = result.data["duplicate_column_pairs"]
    assert any(set(pair) == {"a", "b"} for pair in pairs)


def test_polars_dataframe_handled(tmp_path):
    """Task must handle Polars DataFrames by converting to pandas internally."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "z": [4, 5, 6]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    pairs = result.data["duplicate_column_pairs"]
    assert any(set(pair) == {"x", "y"} for pair in pairs)


def test_no_plots_generated(tmp_path):
    """Duplicate column detection must not generate plots."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
