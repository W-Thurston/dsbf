# tests/eda/test_tasks/test_summarize_dataset_shape.py

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from dsbf.eda.tasks.summarize_dataset_shape import SummarizeDatasetShape
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_shape_reported_correctly(tmp_path) -> None:
    """num_rows and num_columns must match the input DataFrame dimensions."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeDatasetShape)

    assert result.status == "success"
    assert result.data["num_rows"] == 3
    assert result.data["num_columns"] == 3


def test_null_percentage_computed(tmp_path) -> None:
    """null_cell_percentage must reflect actual proportion of null cells."""
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeDatasetShape)

    assert result.status == "success"
    # 2 nulls out of 6 cells = 33.3%
    assert abs(result.data["null_cell_percentage"] - round(2 / 6, 4)) < 1e-4


def test_memory_usage_positive(tmp_path) -> None:
    """
    approx_memory_MB must be non-negative.

    large datasets produce a measurable value.
    """
    # Small DataFrame — rounds to 0.00 MB, which is valid
    df_small = pd.DataFrame({"x": list(range(100))})
    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df_small,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeDatasetShape)
    assert result.status == "success"
    assert result.data["approx_memory_MB"] >= 0

    # Large DataFrame — must show meaningful memory usage
    df_large = pd.DataFrame({"x": list(range(100_000))})
    ctx2, _ = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df_large,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result2: TaskResult = run_task_with_dependencies(ctx2, SummarizeDatasetShape)
    assert result2.data["approx_memory_MB"] > 0


def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must correctly process Polars DataFrames via pandas conversion."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeDatasetShape)

    assert result.status == "success"
    assert result.data["num_rows"] == 3
    assert result.data["num_columns"] == 2


def test_no_plots_generated(tmp_path) -> None:
    """Shape summary must not generate plots."""
    df = pd.DataFrame({"a": [1, 2, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeDatasetShape)

    assert result.status == "success"
    assert result.plots is None
