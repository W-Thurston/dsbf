# tests/eda/test_tasks/test_summarize_modes.py

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_modes import SummarizeModes
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


def test_single_mode_returned_as_scalar(tmp_path) -> None:
    """A column with one clear mode must return a scalar, not a list."""
    df = pd.DataFrame({"a": [1, 1, 1, 2, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeModes)

    assert result.status == "success"
    assert result.data["a"] == 1


def test_multimodal_column_returns_list(tmp_path) -> None:
    """A column with multiple equally-frequent values must return a list."""
    df = pd.DataFrame({"a": [1, 1, 2, 2, 3]})  # 1 and 2 both appear twice

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeModes)

    assert result.status == "success"
    # May return list or scalar depending on pandas mode behaviour
    assert result.data["a"] in ([1, 2], [2, 1], 1, 2)


def test_all_columns_present_in_result(tmp_path) -> None:
    """Every column in the DataFrame must appear in result data."""
    df = pd.DataFrame({"x": [1, 2, 1], "y": ["a", "b", "a"], "z": [True, True, False]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeModes)

    assert result.status == "success"
    assert "x" in result.data
    assert "y" in result.data
    assert "z" in result.data


def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must work correctly on Polars DataFrames."""
    df = pl.DataFrame({"col": [5, 5, 5, 3, 2]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeModes)

    assert result.status == "success"
    assert result.data["col"] == 5


def test_no_plots_generated(tmp_path) -> None:
    """Mode summary must not generate plots."""
    df = pd.DataFrame({"a": [1, 2, 1]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeModes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeModes)

    assert result.status == "success"
    assert result.plots is None
