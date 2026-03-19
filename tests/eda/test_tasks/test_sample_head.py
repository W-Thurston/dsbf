# tests/eda/test_tasks/test_sample_head.py

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from dsbf.eda.tasks.sample_head import SampleHead
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_returns_first_n_rows(tmp_path) -> None:
    """Sample must contain exactly N rows in column-oriented format."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]})

    ctx, task = make_ctx_and_task(
        task_cls=SampleHead,
        current_df=df,
        task_overrides={"n": 3},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    sample = result.data["sample"]
    assert len(sample["a"]) == 3
    assert sample["a"] == [1, 2, 3]


def test_default_n_is_five(tmp_path) -> None:
    """Default n must be 5 when not configured."""
    df = pd.DataFrame({"a": list(range(20))})

    ctx, _ = make_ctx_and_task(
        task_cls=SampleHead,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SampleHead)

    assert result.status == "success"
    assert len(result.data["sample"]["a"]) == 5


def test_n_larger_than_dataset_returns_all_rows(tmp_path) -> None:
    """Requesting more rows than the dataset has must return all rows without error."""
    df = pd.DataFrame({"a": [1, 2, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=SampleHead,
        current_df=df,
        task_overrides={"n": 100},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SampleHead)

    assert result.status == "success"
    assert len(result.data["sample"]["a"]) == 3


def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must work correctly on Polars DataFrames."""
    df = pl.DataFrame({"x": [10, 20, 30, 40, 50]})

    ctx, task = make_ctx_and_task(
        task_cls=SampleHead,
        current_df=df,
        task_overrides={"n": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["sample"]["x"] == [10, 20]


def test_metadata_n_stored(tmp_path) -> None:
    """The n value used must be stored in result metadata."""
    df = pd.DataFrame({"a": list(range(10))})

    ctx, task = make_ctx_and_task(
        task_cls=SampleHead,
        current_df=df,
        task_overrides={"n": 4},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.metadata["n"] == 4


def test_no_plots_generated(tmp_path) -> None:
    """Sample head must not generate plots."""
    df = pd.DataFrame({"a": [1, 2, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=SampleHead,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SampleHead)

    assert result.status == "success"
    assert result.plots is None
