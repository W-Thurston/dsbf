# tests/eda/test_tasks/test_infer_types.py

from typing import Any

import pandas as pd
import polars as pl
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.infer_types import InferTypes
from tests.helpers.context_utils import make_ctx_and_task


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_semantic_type_classifications(tmp_path) -> None:
    """Core semantic type classifications must be correct for each column type."""
    df = pd.DataFrame(
        {
            "numeric": [1, 2, 3],
            "binary": [0, 1, 0],
            "bool_col": [True, False, True],
            "datetime_str": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "short_text": ["a", "b", "c"],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=InferTypes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert result.data["numeric"]["inferred_dtype"] in ("int64", "float64")
    assert result.data["numeric"]["analysis_intent_dtype"] == "continuous"
    assert result.data["binary"]["analysis_intent_dtype"] == "categorical"
    assert result.data["bool_col"]["analysis_intent_dtype"] == "categorical"
    assert result.data["datetime_str"]["analysis_intent_dtype"] == "datetime"
    # Low-cardinality short strings may be classified as id, text, or categorical
    assert result.data["short_text"]["analysis_intent_dtype"] in (
        "id",
        "text",
        "categorical",
    )


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_id_column_classification(tmp_path) -> None:
    """A near-unique string column must be classified as 'id'."""
    df = pd.DataFrame({"user_id": [f"u{i}" for i in range(100)]})

    ctx, task = make_ctx_and_task(
        task_cls=InferTypes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["user_id"]["analysis_intent_dtype"] == "id"


def test_all_null_column_classified_as_unknown(tmp_path) -> None:
    """A column with all null values must be classified as 'unknown'."""
    df = pd.DataFrame({"empty": [None, None, None]})

    ctx, task = make_ctx_and_task(
        task_cls=InferTypes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["empty"]["analysis_intent_dtype"] == "unknown"


def test_metadata_written_to_context(tmp_path) -> None:
    """Semantic types and inferred dtypes must be written to the context metadata."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    ctx, task = make_ctx_and_task(
        task_cls=InferTypes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.run_task(task)

    semantic_types: Any | None = ctx.get_metadata("semantic_types")
    inferred_dtypes: Any | None = ctx.get_metadata("inferred_dtypes")

    assert semantic_types is not None
    assert "a" in semantic_types
    assert "b" in semantic_types
    assert inferred_dtypes is not None
    assert "a" in inferred_dtypes


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must accept Polars DataFrames and convert them internally."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    ctx, task = make_ctx_and_task(
        task_cls=InferTypes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "x" in result.data
    assert "y" in result.data
    assert result.data["x"]["analysis_intent_dtype"] == "continuous"


def test_no_plots_generated(tmp_path) -> None:
    """infer_types must not generate any plots."""
    df = pd.DataFrame({"a": [1, 2, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=InferTypes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_low_cardinality_numeric_classified_as_categorical(tmp_path) -> None:
    """A numeric column with ≤ 20 unique values and < 5% unique ratio is categorical."""
    df = pd.DataFrame({"rating": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] * 10})

    ctx, task = make_ctx_and_task(
        task_cls=InferTypes,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["rating"]["analysis_intent_dtype"] == "categorical"
