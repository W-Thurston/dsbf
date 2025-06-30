# tests/eda/test_tasks/test_suggest_categorical_encoding.py

import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.suggest_categorical_encoding import SuggestCategoricalEncoding
from tests.helpers.context_utils import make_ctx_and_task


def test_suggest_one_hot_and_frequency():
    df = pl.DataFrame(
        {
            "low_card": ["a", "b", "c", "a", "b"] * 20,  # 100 rows
            "mid_card": [f"val_{i % 20}" for i in range(100)],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SuggestCategoricalEncoding,
        current_df=df,
        task_overrides={
            "low_cardinality_threshold": 5,
            "high_cardinality_threshold": 30,
        },
    )

    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    suggestions = result.data["encoding_suggestions"]
    assert suggestions["low_card"]["suggested_encoding"].startswith("one-hot")
    assert suggestions["mid_card"]["suggested_encoding"].startswith("frequency")


def test_suggest_high_cardinality_tagging():
    df = pl.DataFrame(
        {
            "high_card": [f"user_{i}" for i in range(100)],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SuggestCategoricalEncoding,
        current_df=df,
        task_overrides={
            "low_cardinality_threshold": 5,
            "high_cardinality_threshold": 30,
        },
    )

    result = ctx.run_task(task)

    assert result.data is not None
    suggestions = result.data["encoding_suggestions"]
    assert suggestions["high_card"]["cardinality"] == 100
    assert "high-cardinality" in suggestions["high_card"]["suggested_encoding"]


def test_target_encoding_is_suggested_for_numeric_target():
    df = pl.DataFrame(
        {
            "cat": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
            "target": [1] * 20 + [5] * 20 + [10] * 20,
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=SuggestCategoricalEncoding,
        current_df=df,
        task_overrides={
            "low_cardinality_threshold": 2,
            "high_cardinality_threshold": 10,
            "target_column": "target",
        },
    )

    result = ctx.run_task(task)

    assert result.data is not None
    encoding = result.data["encoding_suggestions"]["cat"]["suggested_encoding"]
    assert "target encoding" in encoding


def test_missing_target_column_is_gracefully_handled():
    df = pl.DataFrame({"color": ["red", "blue", "red", "green"] * 5})

    ctx, task = make_ctx_and_task(
        task_cls=SuggestCategoricalEncoding,
        current_df=df,
        task_overrides={"target_column": "missing_target"},
    )

    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert (
        "target encoding"
        not in result.data["encoding_suggestions"]["color"]["suggested_encoding"]
    )
