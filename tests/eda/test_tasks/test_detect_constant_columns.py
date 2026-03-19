# tests/eda/test_tasks/test_detect_constant_columns.py

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from dsbf.eda.tasks.detect_constant_columns import DetectConstantColumns
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_constant_columns_detected(tmp_path) -> None:
    """Columns with exactly one unique value must be identified."""
    df = pd.DataFrame(
        {
            "a": [1, 1, 1],  # constant int
            "b": [1, 2, 3],  # varying
            "c": ["x", "x", "x"],  # constant str
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectConstantColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert sorted(result.data["constant_columns"]) == ["a", "c"]
    assert "b" not in result.data["constant_columns"]


def test_no_constant_columns(tmp_path) -> None:
    """A dataset with no constant columns must return an empty list."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectConstantColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["constant_columns"] == []


def test_guidance_attached_for_constant_columns(tmp_path) -> None:
    """EDA and ML guidance blurbs must be attached for every constant column."""
    df = pd.DataFrame({"const": [0, 0, 0, 0, 0], "vary": [1, 2, 3, 4, 5]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectConstantColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["constant_columns"] == ["const"]
    assert result.guidance is not None
    assert "const" in result.guidance
    assert len(result.guidance["const"]["eda"]) > 0
    assert len(result.guidance["const"]["ml"]) > 0
    assert result.guidance["const"]["eda"][0]["level"] == "error"
    drop_actions = result.guidance["const"]["ml"][0]["actions"]
    assert any(a["action"] == "drop" for a in drop_actions)


def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must work correctly on Polars DataFrames."""
    df = pl.DataFrame(
        {
            "fixed": [99, 99, 99],
            "varies": [1, 2, 3],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectConstantColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "fixed" in result.data["constant_columns"]
    assert "varies" not in result.data["constant_columns"]


def test_metadata_engine_field_set_correctly(tmp_path) -> None:
    """Metadata engine field must reflect the actual backend used."""
    df_pd = pd.DataFrame({"a": [1, 2, 3]})
    df_pl = pl.DataFrame({"a": [1, 2, 3]})

    for df, expected_engine in [(df_pd, "pandas"), (df_pl, "polars")]:
        ctx, task = make_ctx_and_task(
            task_cls=DetectConstantColumns,
            current_df=df,
            global_overrides={"output_dir": str(tmp_path)},
        )
        result: TaskResult = ctx.run_task(task)
        assert result.metadata["engine"] == expected_engine


def test_no_plots_generated(tmp_path) -> None:
    """Constant column detection must not generate plots."""
    df = pd.DataFrame({"c": [1, 1, 1], "v": [1, 2, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectConstantColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
