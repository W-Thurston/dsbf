# tests/eda/test_tasks/test_detect_duplicates.py

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_duplicates import DetectDuplicates
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


def test_duplicate_rows_counted_correctly(tmp_path):
    """Duplicate rows must be counted as non-first occurrences."""
    df = pd.DataFrame(
        {
            "a": [1, 1, 2, 3, 3],
            "b": ["x", "x", "y", "z", "z"],  # rows 0==1, rows 3==4
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDuplicates,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDuplicates)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data["duplicate_count"] == 2


def test_no_duplicates_returns_zero(tmp_path):
    """A dataset with no duplicate rows must return duplicate_count of 0."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDuplicates,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDuplicates)

    assert result.status == "success"
    assert result.data["duplicate_count"] == 0


def test_guidance_emitted_when_duplicates_present(tmp_path):
    """EDA guidance blurb under '__dataset__' must be emitted when duplicates exist."""
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDuplicates,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDuplicates)

    assert result.status == "success"
    assert result.data["duplicate_count"] == 1
    assert result.guidance is not None
    assert "__dataset__" in result.guidance
    assert len(result.guidance["__dataset__"]["eda"]) > 0


def test_no_guidance_when_no_duplicates(tmp_path):
    """No guidance must be emitted when no duplicate rows exist."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDuplicates,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDuplicates)

    assert result.status == "success"
    assert result.guidance is None or "__dataset__" not in (result.guidance or {})


def test_polars_dataframe_handled(tmp_path):
    """Task must handle Polars DataFrames using .unique() for count derivation."""
    df = pl.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDuplicates,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDuplicates)

    assert result.status == "success"
    assert result.data["duplicate_count"] == 1


def test_no_plots_generated(tmp_path):
    """Duplicate row detection must not generate plots."""
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDuplicates,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDuplicates)

    assert result.status == "success"
    assert result.plots is None
