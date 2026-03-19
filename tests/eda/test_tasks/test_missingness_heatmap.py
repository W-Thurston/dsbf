# tests/eda/test_tasks/test_missingness_heatmap.py

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from dsbf.core.context import AnalysisContext
from dsbf.eda.tasks.missingness_heatmap import MissingnessHeatmap
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_missing_cell_count_correct(tmp_path) -> None:
    """Total missing cell count must match the sum of nulls across all columns."""
    df = pd.DataFrame({"a": [1, None], "b": [None, 2]})

    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    result: TaskResult = ctx.run_task(MissingnessHeatmap())

    assert result.status == "success"
    assert result.data["missing_cells"] == 2


def test_missing_columns_identified(tmp_path) -> None:
    """Columns with at least one null must appear in missing_columns."""
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3], "c": [None, None, None]})

    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    result: TaskResult = ctx.run_task(MissingnessHeatmap())

    assert result.status == "success"
    assert "a" in result.data["missing_columns"]
    assert "c" in result.data["missing_columns"]
    assert "b" not in result.data["missing_columns"]


def test_shape_recorded(tmp_path) -> None:
    """column_count and row_count must reflect the dataset dimensions."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    result: TaskResult = ctx.run_task(MissingnessHeatmap())

    assert result.status == "success"
    assert result.data["column_count"] == 2
    assert result.data["row_count"] == 3


def test_no_missing_data(tmp_path) -> None:
    """Dataset must report zero missing cells + empty missing_columns list."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    result: TaskResult = ctx.run_task(MissingnessHeatmap())

    assert result.status == "success"
    assert result.data["missing_cells"] == 0
    assert result.data["missing_columns"] == []


def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must accept Polars DataFrames via the backend conversion path."""
    df = pl.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=MissingnessHeatmap,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, MissingnessHeatmap)

    assert result.status == "success"
    assert result.data["missing_cells"] == 2


def test_no_plots_generated(tmp_path) -> None:
    """MissingnessHeatmap is a summary-only task and must not generate plots."""
    df = pd.DataFrame({"a": [1, None], "b": [None, 2]})

    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    result: TaskResult = ctx.run_task(MissingnessHeatmap())

    assert result.status == "success"
    assert result.plots is None
