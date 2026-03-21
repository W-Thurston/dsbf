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
    assert abs(result.data["null_cell_percentage"] - round(2 / 6, 4)) < 1e-4


def test_memory_usage_aggregate(tmp_path) -> None:
    """
    approx_memory_MB must be non-negative.

    large datasets produce a measurable value.
    """
    df_small = pd.DataFrame({"x": list(range(100))})
    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df_small,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeDatasetShape)
    assert result.status == "success"
    assert result.data["approx_memory_MB"] >= 0

    df_large = pd.DataFrame({"x": list(range(100_000))})
    ctx2, _ = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df_large,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result2: TaskResult = run_task_with_dependencies(ctx2, SummarizeDatasetShape)
    assert result2.data["approx_memory_MB"] > 0


def test_per_column_memory_bytes_present(tmp_path) -> None:
    """column_memory_bytes must contain an entry for every column."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.0, 2.0, 3.0]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeDatasetShape)

    assert result.status == "success"
    col_mem = result.data["column_memory_bytes"]
    assert set(col_mem.keys()) == {"a", "b", "c"}
    assert all(isinstance(v, int) and v >= 0 for v in col_mem.values())


def test_per_column_memory_mb_matches_bytes(tmp_path) -> None:
    """column_memory_MB must be the byte values divided by 1_048_576."""
    df = pd.DataFrame({"x": list(range(1000)), "y": [float(i) for i in range(1000)]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeDatasetShape)

    assert result.status == "success"
    for col in ("x", "y"):
        bytes_val = result.data["column_memory_bytes"][col]
        mb_val = result.data["column_memory_MB"][col]
        assert abs(mb_val - round(bytes_val / 1_048_576, 4)) < 1e-6


def test_object_column_deep_memory_counted(tmp_path) -> None:
    """Object dtype columns must use deep=True memory est (strings are larger)."""
    df = pd.DataFrame({"text": ["a" * 1000] * 100})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeDatasetShape)

    assert result.status == "success"
    # 100 strings of 1000 chars each — deep memory must be substantially larger than
    # the shallow pointer-only estimate of 100 * 8 = 800 bytes.
    assert result.data["column_memory_bytes"]["text"] > 800


def test_aggregate_memory_equals_sum_of_columns(tmp_path) -> None:
    """approx_memory_MB must be consistent with the sum of column_memory_bytes."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeDatasetShape)

    assert result.status == "success"
    # The aggregate includes the pandas Index entry which is excluded from
    # column_memory_bytes — so aggregate >= sum of columns.
    col_sum_mb = sum(result.data["column_memory_MB"].values())
    assert result.data["approx_memory_MB"] >= col_sum_mb - 0.001  # float tolerance


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
    assert "a" in result.data["column_memory_bytes"]
    assert "b" in result.data["column_memory_bytes"]


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
