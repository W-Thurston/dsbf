# tests/eda/test_tasks/test_sample_tail.py

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from dsbf.eda.tasks.sample_tail import SampleTail
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_returns_last_n_rows(tmp_path) -> None:
    """Sample must contain the last N rows in column-oriented format."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]})

    ctx, task = make_ctx_and_task(
        task_cls=SampleTail,
        current_df=df,
        task_overrides={"n": 3},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    sample = result.data["sample"]
    assert len(sample["a"]) == 3
    assert sample["a"] == [3, 4, 5]


def test_default_n_is_five(tmp_path) -> None:
    """Default n must be 5 when not configured."""
    df = pd.DataFrame({"a": list(range(20))})

    ctx, _ = make_ctx_and_task(
        task_cls=SampleTail,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SampleTail)

    assert result.status == "success"
    assert len(result.data["sample"]["a"]) == 5


def test_n_zero_returns_empty_sample(tmp_path) -> None:
    """n=0 must return empty sample dict rather than raising or returning all rows."""
    df = pd.DataFrame({"a": [1, 2, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=SampleTail,
        current_df=df,
        task_overrides={"n": 0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["sample"]["a"] == []


def test_n_larger_than_dataset_returns_all_rows(tmp_path) -> None:
    """Requesting more rows than the dataset has must return all rows without error."""
    df = pd.DataFrame({"a": [1, 2, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=SampleTail,
        current_df=df,
        task_overrides={"n": 100},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SampleTail)

    assert result.status == "success"
    assert len(result.data["sample"]["a"]) == 3


def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must work correctly on Polars DataFrames."""
    df = pl.DataFrame({"x": [10, 20, 30, 40, 50]})

    ctx, task = make_ctx_and_task(
        task_cls=SampleTail,
        current_df=df,
        task_overrides={"n": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["sample"]["x"] == [40, 50]


def test_metadata_n_stored(tmp_path) -> None:
    """The n value used must be stored in result metadata."""
    df = pd.DataFrame({"a": list(range(10))})

    ctx, task = make_ctx_and_task(
        task_cls=SampleTail,
        current_df=df,
        task_overrides={"n": 4},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.metadata["n"] == 4


def test_no_plots_generated(tmp_path) -> None:
    """Sample tail must not generate plots."""
    df = pd.DataFrame({"a": [1, 2, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=SampleTail,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SampleTail)

    assert result.status == "success"
    assert result.plots is None
