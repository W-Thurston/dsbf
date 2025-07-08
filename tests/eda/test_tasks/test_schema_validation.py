# tests/eda/test_tasks/test_schema_validation.py

import polars as pl
import pytest

from dsbf.eda.tasks.schema_validation import SchemaValidation
from tests.helpers.context_utils import make_ctx_and_task


@pytest.fixture
def titanic_like_df():
    return pl.DataFrame(
        {
            "survived": [1, 0, 1],
            "pclass": [3, 1, 2],
            "sex": ["male", "female", "female"],
            "age": [22.0, 38.0, 26.0],
            "fare": [7.25, 71.28, 7.92],
            "embarked": ["S", "C", "Q"],
        }
    )


def test_validation_skipped(titanic_like_df):
    ctx, task = make_ctx_and_task(
        task_cls=SchemaValidation,
        current_df=titanic_like_df,
        global_overrides={
            "schema_validation": {
                "enable_schema_validation": False,
            }
        },
    )
    result = ctx.run_task(task)

    assert result.status == "skipped"
    assert "disabled" in result.summary["message"].lower()


def test_validation_passes_clean(titanic_like_df):
    ctx, task = make_ctx_and_task(
        task_cls=SchemaValidation,
        current_df=titanic_like_df,
        global_overrides={
            "schema_validation": {
                "enable_schema_validation": True,
                "fail_or_warn": "warn",
                "schema": {
                    "required_columns": ["survived", "sex"],
                    "dtypes": {"age": "float", "sex": "str"},
                    "value_ranges": {"fare": {"min": 0}},
                    "categories": {"sex": ["male", "female"]},
                },
            },
        },
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert (
        result.reliability_warnings is None
        or "warning" not in result.reliability_warnings
    )


def test_missing_required_columns(titanic_like_df):
    ctx, task = make_ctx_and_task(
        task_cls=SchemaValidation,
        current_df=titanic_like_df,
        global_overrides={
            "schema_validation": {
                "enable_schema_validation": True,
                "fail_or_warn": "warn",
                "schema": {"required_columns": ["nonexistent"]},
            },
        },
    )
    result = ctx.run_task(task)

    assert result.data is not None
    assert "nonexistent" in result.data["missing_columns"]
    assert result.status == "success"
    assert result.reliability_warnings is not None
    assert "missing_col_nonexistent" in result.reliability_warnings["error"]


def test_dtype_mismatch(titanic_like_df):
    ctx, task = make_ctx_and_task(
        task_cls=SchemaValidation,
        current_df=titanic_like_df,
        global_overrides={
            "schema_validation": {
                "enable_schema_validation": True,
                "schema": {"dtypes": {"pclass": "str"}},
            },
        },
    )
    result = ctx.run_task(task)

    assert result.data is not None
    assert "pclass" in result.data["dtype_mismatches"]


def test_value_out_of_range(titanic_like_df):
    df = titanic_like_df.with_columns(pl.col("age") + 200)
    ctx, task = make_ctx_and_task(
        task_cls=SchemaValidation,
        current_df=df,
        global_overrides={
            "schema_validation": {
                "enable_schema_validation": True,
                "schema": {"value_ranges": {"age": {"max": 100}}},
            },
        },
    )
    result = ctx.run_task(task)

    assert result.data is not None
    assert "age" in result.data["value_range_violations"]


def test_unexpected_category(titanic_like_df):
    df = titanic_like_df.with_columns(pl.lit("other").alias("sex"))
    ctx, task = make_ctx_and_task(
        task_cls=SchemaValidation,
        current_df=df,
        global_overrides={
            "schema_validation": {
                "enable_schema_validation": True,
                "schema": {"categories": {"sex": ["male", "female"]}},
            },
        },
    )
    result = ctx.run_task(task)

    assert result.data is not None
    assert "sex" in result.data["unexpected_categories"]


def test_fail_mode_blocks_pipeline(titanic_like_df):
    ctx, task = make_ctx_and_task(
        task_cls=SchemaValidation,
        current_df=titanic_like_df,
        global_overrides={
            "schema_validation": {
                "enable_schema_validation": True,
                "fail_or_warn": "fail",
                "schema": {"required_columns": ["missing_col"]},
            },
        },
    )
    result = ctx.run_task(task)

    assert result.status == "failed"


def test_unknown_schema_key_warning(titanic_like_df):
    ctx, task = make_ctx_and_task(
        task_cls=SchemaValidation,
        current_df=titanic_like_df,
        global_overrides={
            "schema_validation": {
                "enable_schema_validation": True,
                "schema": {"required": ["survived"]},  # invalid key
            },
        },
    )
    result = ctx.run_task(task)

    assert result.reliability_warnings is not None
    assert "unknown_schema_key_required" in result.reliability_warnings["warning"]
