# tests/eda/test_tasks/test_detect_duplicate_columns.py

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_duplicate_columns import DetectDuplicateColumns
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


def test_duplicate_pair_detected(tmp_path):
    """An exact duplicate column pair must appear in duplicate_column_pairs."""
    # Use 30 rows so unique_ratio stays below the 0.9 ID-detection
    # threshold — with only 3 rows, all-unique integer columns get
    # classified as ID-like and excluded from get_columns_by_intent().
    import numpy as np

    rng = np.random.default_rng(42)
    base = rng.integers(1, 10, size=30).tolist()
    df = pd.DataFrame(
        {
            "a": base,
            "b": base,  # exact duplicate of a
            "c": rng.integers(1, 10, size=30).tolist(),
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDuplicateColumns)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    pairs = result.data["duplicate_column_pairs"]
    assert any(set(pair) == {"a", "b"} for pair in pairs)
    # c is distinct - must not appear in any pair
    assert not any("c" in pair for pair in pairs)


def test_no_duplicate_columns(tmp_path):
    """A dataset with no duplicate columns must return an empty list."""
    # Use repeated values to avoid ID-like classification
    df = pd.DataFrame(
        {
            "x": list(range(5)) * 6,
            "y": list(range(5, 10)) * 6,
            "z": list(range(10, 15)) * 6,
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDuplicateColumns)

    assert result.status == "success"
    assert result.data["duplicate_column_pairs"] == []


def test_guidance_attached_for_duplicate_pair(tmp_path):
    """EDA and ML guidance blurbs must be attached for duplicate columns."""
    # Use repeated values to avoid ID-like classification
    vals: list[int] = list(range(1, 6)) * 6
    df = pd.DataFrame({"a": vals, "b": vals})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDuplicateColumns)

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
    # Use repeated values to avoid ID-like classification
    vals: list[int | None] = [1, None, 3] * 10
    df = pd.DataFrame({"a": vals, "b": vals})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDuplicateColumns)

    assert result.status == "success"
    pairs = result.data["duplicate_column_pairs"]
    assert any(set(pair) == {"a", "b"} for pair in pairs)


def test_polars_dataframe_handled(tmp_path):
    """Task must handle Polars DataFrames by converting to pandas internally."""
    # Use repeated values to avoid ID-like classification
    vals: list[int] = list(range(1, 6)) * 6
    df = pl.DataFrame({"x": vals, "y": vals, "z": list(range(6, 11)) * 6})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDuplicateColumns)

    assert result.status == "success"
    pairs = result.data["duplicate_column_pairs"]
    assert any(set(pair) == {"x", "y"} for pair in pairs)


def test_no_plots_generated(tmp_path):
    """Duplicate column detection must not generate plots."""
    vals: list[int] = list(range(1, 6)) * 6
    df = pd.DataFrame({"a": vals, "b": vals})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDuplicateColumns)

    assert result.status == "success"
    assert result.plots is None
