# tests/eda/test_tasks/test_detect_mixed_type_columns.py

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_mixed_type_columns import DetectMixedTypeColumns
from tests.helpers.context_utils import make_ctx_and_task


def test_all_same_type_column_not_flagged(tmp_path) -> None:
    """A uniformly typed column must not be flagged."""
    df = pl.DataFrame({"col1": [1, 2, 3, 4, 5]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectMixedTypeColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.summary["num_mixed_type_columns"] == 0
    assert result.summary["columns"] == []


def test_detects_mixed_type_column(tmp_path) -> None:
    """A column containing both int and str values must be flagged."""
    df = pl.DataFrame(
        {"col1": pl.Series("col1", [1, 2, "three", 4.0, None], dtype=pl.Object)}
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectMixedTypeColumns,
        current_df=df,
        task_overrides={"min_ratio": 0.1, "ignore_null_type": True},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_mixed_type_columns"] == 1
    assert "col1" in result.summary["columns"]
    assert "str" in result.data["col1"]["type_counts"]


def test_ignores_minor_type_below_threshold(tmp_path) -> None:
    """A minority type representing less than min_ratio must not trigger a flag."""
    df = pl.DataFrame(
        {"col1": pl.Series("col1", [1] * 98 + ["x"] * 2, dtype=pl.Object)},
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectMixedTypeColumns,
        current_df=df,
        task_overrides={"min_ratio": 0.05, "ignore_null_type": True},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_mixed_type_columns"] == 0


def test_null_type_excluded_when_ignored(tmp_path) -> None:
    """NoneType must not appear in type_counts when ignore_null_type=True."""
    df = pl.DataFrame({"col1": pl.Series("col1", [1, "two", 3, None], dtype=pl.Object)})

    ctx, task = make_ctx_and_task(
        task_cls=DetectMixedTypeColumns,
        current_df=df,
        task_overrides={"min_ratio": 0.1, "ignore_null_type": True},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_mixed_type_columns"] == 1
    assert "NoneType" not in result.data["col1"]["type_counts"]


def test_skips_strictly_typed_column(tmp_path) -> None:
    """A non-Object Polars column cannot have mixed types and must not be flagged."""
    df = pl.DataFrame({"col1": [1, 2, 3, 4]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectMixedTypeColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_mixed_type_columns"] == 0


def test_guidance_attached_for_mixed_columns(tmp_path) -> None:
    """An EDA guidance blurb must be attached for each mixed-type column."""
    df = pl.DataFrame(
        {"mixed": pl.Series("mixed", [1, 2, "three", 4, "five"] * 10, dtype=pl.Object)},
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectMixedTypeColumns,
        current_df=df,
        task_overrides={"min_ratio": 0.1, "ignore_null_type": True},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_mixed_type_columns"] == 1
    assert result.guidance is not None
    assert "mixed" in result.guidance
    assert len(result.guidance["mixed"]["eda"]) > 0


def test_pandas_dataframe_handled(tmp_path) -> None:
    """Task must detect mixed types in pandas object dtype columns."""
    df = pd.DataFrame(
        {"col": pd.array([1, 2, "three", 4.0, "five"] * 10, dtype=object)},
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectMixedTypeColumns,
        current_df=df,
        task_overrides={"min_ratio": 0.1, "ignore_null_type": True},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_mixed_type_columns"] == 1


def test_no_plots_generated(tmp_path) -> None:
    """Mixed-type column detection must not generate plots."""
    df = pl.DataFrame(
        {"col1": pl.Series("col1", [1, 2, "three", 4.0, None], dtype=pl.Object)},
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectMixedTypeColumns,
        current_df=df,
        task_overrides={"min_ratio": 0.1},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
