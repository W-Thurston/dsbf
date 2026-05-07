# tests/eda/test_tasks/test_summarize_value_counts.py

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from dsbf.eda.tasks.summarize_value_counts import SummarizeValueCounts
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_top_k_values_returned(tmp_path) -> None:
    """Result must contain no more than top_k entries per column."""
    df = pd.DataFrame({"a": list("abcde") * 4})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeValueCounts,
        current_df=df,
        task_overrides={"top_k": 3},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeValueCounts)

    assert result.status == "success"
    assert len(result.data["a"]) == 3


def test_all_columns_summarized(tmp_path) -> None:
    """Every column in the DataFrame must appear in result data."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "a"]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeValueCounts,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeValueCounts)

    assert result.status == "success"
    assert "x" in result.data
    assert "y" in result.data


def test_polars_dataframe_handled(tmp_path) -> None:
    df = pl.DataFrame({"col": ["A", "B", "A", "C", "A"]})
    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeValueCounts,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeValueCounts)
    assert result.status == "success"
    assert "col" in result.data


def test_metadata_top_k_stored(tmp_path) -> None:
    """The top_k value must be stored in result metadata."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeValueCounts,
        current_df=df,
        task_overrides={"top_k": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeValueCounts)
    assert result.status == "success"
    assert result.metadata["top_k"] == 2


def test_no_plots_generated(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeValueCounts,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeValueCounts)
    assert result.status == "success"
    assert result.plots is None
