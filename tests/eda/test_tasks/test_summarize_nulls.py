# tests/eda/test_tasks/test_summarize_nulls.py

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from dsbf.eda.tasks.summarize_nulls import SummarizeNulls
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_null_counts_and_percentages_correct(tmp_path) -> None:
    """Null counts and percentages must reflect the actual missing values."""
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNulls,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNulls)

    assert result.status == "success"
    assert result.data["null_counts"]["a"] == 1
    assert result.data["null_counts"]["b"] == 2
    assert abs(result.data["null_percentages"]["a"] - 1 / 3) < 1e-4
    assert abs(result.data["null_percentages"]["b"] - 2 / 3) < 1e-4


def test_high_null_columns_identified(tmp_path) -> None:
    """Columns exceeding the null_threshold must appear in high_null_columns."""
    df = pd.DataFrame({"mostly_null": [None] * 8 + [1, 2], "clean": list(range(10))})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNulls,
        current_df=df,
        task_overrides={"null_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNulls)

    assert result.status == "success"
    assert "mostly_null" in result.data["high_null_columns"]
    assert "clean" not in result.data["high_null_columns"]


def test_guidance_attached_for_missing_columns(tmp_path) -> None:
    """EDA and ML guidance must be emitted for columns with ≥ 5% missing."""
    df = pd.DataFrame({"a": [None] * 10 + list(range(90))})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNulls,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNulls)

    assert result.status == "success"
    assert result.guidance is not None
    assert "a" in result.guidance
    assert len(result.guidance["a"]["eda"]) > 0
    assert len(result.guidance["a"]["ml"]) > 0


def test_no_guidance_for_clean_columns(tmp_path) -> None:
    """A complete column must not emit guidance."""
    df = pd.DataFrame({"clean": [1, 2, 3, 4, 5]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNulls,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNulls)

    assert result.status == "success"
    assert result.guidance is None or "clean" not in (result.guidance or {})


def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    df = pl.DataFrame({"a": [1, None, 3, None, 5]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNulls,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNulls)

    assert result.status == "success"
    assert result.data["null_counts"]["a"] == 2


def test_no_plots_generated(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, None, 3]})
    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNulls,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNulls)
    assert result.status == "success"
    assert result.plots is None
