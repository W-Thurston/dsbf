# tests/eda/test_tasks/test_summarize_unique.py

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from dsbf.eda.tasks.summarize_unique import SummarizeUnique
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_unique_counts_correct(tmp_path) -> None:
    """Unique count must equal the number of distinct values per column."""
    df = pd.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "x", "x", "x"]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeUnique,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeUnique)

    assert result.status == "success"
    assert result.data["a"] == 3
    assert result.data["b"] == 1


def test_all_columns_present(tmp_path) -> None:
    """Every column must appear in result data."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeUnique,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeUnique)

    assert result.status == "success"
    assert "x" in result.data
    assert "y" in result.data


def test_polars_dataframe_handled(tmp_path) -> None:
    df = pl.DataFrame({"col": [1, 2, 2, 3, 3, 3]})
    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeUnique,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeUnique)
    assert result.status == "success"
    assert result.data["col"] == 3


def test_no_plots_generated(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeUnique,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeUnique)
    assert result.status == "success"
    assert result.plots is None
