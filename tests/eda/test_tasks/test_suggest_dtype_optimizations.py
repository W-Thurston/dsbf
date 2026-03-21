# tests/eda/test_tasks/test_suggest_dtype_optimizations.py

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
import pytest

# ── Unit tests for pure downcast rule functions ───────────────────────────────
from dsbf.eda.tasks.suggest_dtype_optimizations import (
    SuggestDtypeOptimizations,
    _fits_bool,
    _fits_category,
    _fits_float32,
    _fits_int8,
    _fits_int16,
    _fits_int32,
)
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_fits_int8_within_range() -> None:
    s = pd.Series([0, 1, 2, 100, -10], dtype="int64")
    assert _fits_int8(s, "int64", "continuous") is True


def test_fits_int8_outside_range() -> None:
    s = pd.Series([0, 128, 200], dtype="int64")
    assert _fits_int8(s, "int64", "continuous") is False


def test_fits_int16_within_range() -> None:
    s = pd.Series([0, 1000, -1000, 32000], dtype="int64")
    assert _fits_int16(s, "int64", "continuous") is True


def test_fits_int16_outside_range() -> None:
    s = pd.Series([0, 40000], dtype="int64")
    assert _fits_int16(s, "int64", "continuous") is False


def test_fits_int32_within_range() -> None:
    s = pd.Series([0, 2_000_000_000, -2_000_000_000], dtype="int64")
    assert _fits_int32(s, "int64", "continuous") is True


def test_fits_float32_safe_range() -> None:
    s = pd.Series([0.1, 1.5, -3.14, 1000.0], dtype="float64")
    assert _fits_float32(s, "float64", "continuous") is True


def test_fits_float32_extreme_values() -> None:
    s = pd.Series([1e38, 2e38], dtype="float64")
    assert _fits_float32(s, "float64", "continuous") is False


def test_fits_bool_zero_one() -> None:
    s = pd.Series([0, 1, 0, 1, 0], dtype="int64")
    assert _fits_bool(s, "int64", "categorical") is True


def test_fits_bool_true_false_strings() -> None:
    s = pd.Series(["true", "false", "true"], dtype="object")
    assert _fits_bool(s, "object", "categorical") is True


def test_fits_bool_more_than_two_values() -> None:
    s = pd.Series([0, 1, 2], dtype="int64")
    assert _fits_bool(s, "int64", "categorical") is False


def test_fits_category_object_categorical() -> None:
    s = pd.Series(["A", "B", "A", "C"], dtype="object")
    assert _fits_category(s, "object", "categorical") is True


def test_fits_category_not_for_continuous() -> None:
    s = pd.Series(["A", "B", "A", "C"], dtype="object")
    assert _fits_category(s, "object", "continuous") is False


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_int64_small_range_suggested_as_int8(tmp_path) -> None:
    df = pd.DataFrame({"small_int": pd.array(list(range(-10, 91)), dtype="int64")})

    ctx, task = make_ctx_and_task(
        task_cls=SuggestDtypeOptimizations,
        current_df=df,
        task_overrides={"min_savings_bytes": 0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"small_int": "continuous"})
    ctx.set_metadata("inferred_dtypes", {"small_int": "int64"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "small_int" in result.data["suggestions"]
    assert result.data["suggestions"]["small_int"]["suggested_dtype"] == "int8"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_float64_safe_range_suggested_as_float32(tmp_path) -> None:
    """A float64 column with safe values must be suggested as float32."""
    df = pd.DataFrame({"measurement": [1.5, 2.3, 0.7, 100.1] * 100})

    ctx, task = make_ctx_and_task(
        task_cls=SuggestDtypeOptimizations,
        current_df=df,
        task_overrides={"min_savings_bytes": 0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"measurement": "continuous"})
    ctx.set_metadata("inferred_dtypes", {"measurement": "float64"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    suggestions = result.data["suggestions"]
    assert "measurement" in suggestions
    assert suggestions["measurement"]["suggested_dtype"] == "float32"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_low_cardinality_object_suggested_as_category(tmp_path) -> None:
    """A low-cardinality object column must be suggested as category."""
    df = pd.DataFrame({"status": ["active", "inactive", "pending"] * 100})

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestDtypeOptimizations,
        current_df=df,
        task_overrides={"min_savings_bytes": 0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestDtypeOptimizations)

    assert result.status == "success"
    suggestions = result.data["suggestions"]
    assert "status" in suggestions
    assert suggestions["status"]["suggested_dtype"] == "category"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_binary_int_column_suggested_as_bool(tmp_path) -> None:
    """An int64 column with only 0 and 1 values must be suggested as bool."""
    df = pd.DataFrame({"flag": pd.array([0, 1, 0, 1, 1, 0] * 100, dtype="int64")})

    ctx, task = make_ctx_and_task(
        task_cls=SuggestDtypeOptimizations,
        current_df=df,
        task_overrides={"min_savings_bytes": 0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    suggestions = result.data["suggestions"]
    assert "flag" in suggestions
    assert suggestions["flag"]["suggested_dtype"] == "bool"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_already_optimal_column_not_suggested(tmp_path) -> None:
    """A column already using an optimal dtype must not appear in suggestions."""
    df = pd.DataFrame({"tiny": pd.array([0, 1, 2, 3], dtype="int8")})

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestDtypeOptimizations,
        current_df=df,
        task_overrides={"min_savings_bytes": 0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestDtypeOptimizations)

    assert result.status == "success"
    assert "tiny" not in result.data["suggestions"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_min_savings_threshold_filters_small_columns(tmp_path) -> None:
    """Columns with savings below min_savings_bytes must not be suggested."""
    # 4 rows of int64 — savings = 4 * (8-1) = 28 bytes, below default 1024
    df = pd.DataFrame({"tiny_int": pd.array([0, 1, 2, 3], dtype="int64")})

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestDtypeOptimizations,
        current_df=df,
        task_overrides={"min_savings_bytes": 1024},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestDtypeOptimizations)

    assert result.status == "success"
    assert "tiny_int" not in result.data["suggestions"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_attached_for_each_suggestion(tmp_path) -> None:
    """An EDA guidance blurb must be attached for each column with a suggestion."""
    df = pd.DataFrame({"small_int": pd.array(list(range(100)), dtype="int64")})

    ctx, task = make_ctx_and_task(
        task_cls=SuggestDtypeOptimizations,
        current_df=df,
        task_overrides={"min_savings_bytes": 0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"small_int": "continuous"})
    ctx.set_metadata("inferred_dtypes", {"small_int": "int64"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    assert "small_int" in result.guidance
    assert len(result.guidance["small_int"]["eda"]) > 0
    action = result.guidance["small_int"]["eda"][0]["actions"][0]
    assert action["action"] == "downcast"
    assert action["from_dtype"] == "int64"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_correct(tmp_path) -> None:
    """suggestion_count in summary must equal the number of suggestions in data."""
    df = pd.DataFrame(
        {
            "a": pd.array(list(range(1000)), dtype="int64"),  # → int16
            "b": [1.5, 2.3] * 500,  # → float32
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestDtypeOptimizations,
        current_df=df,
        task_overrides={"min_savings_bytes": 0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestDtypeOptimizations)

    assert result.status == "success"
    assert result.summary["suggestion_count"] == len(result.data["suggestions"])


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    df = pl.DataFrame({"x": list(range(1000))})

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestDtypeOptimizations,
        current_df=df,
        task_overrides={"min_savings_bytes": 0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestDtypeOptimizations)

    assert result.status == "success"


def test_no_plots_generated(tmp_path) -> None:
    """Dtype optimization task must not generate plots."""
    df = pd.DataFrame({"a": pd.array(list(range(100)), dtype="int64")})

    ctx, _ = make_ctx_and_task(
        task_cls=SuggestDtypeOptimizations,
        current_df=df,
        task_overrides={"min_savings_bytes": 0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SuggestDtypeOptimizations)

    assert result.status == "success"
    assert result.plots is None


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_zero_min_savings_allows_small_columns(tmp_path) -> None:
    """
    Setting min_savings_bytes=0 must allow suggestions even for very small columns.

    This specifically guards against the bug where `0` was treated as falsy
    and replaced with the default (1024), incorrectly filtering out valid
    suggestions.
    """
    # Very small column: savings will be < 1024 bytes
    df = pd.DataFrame({"tiny_int": pd.array([0, 1, 2, 3], dtype="int64")})

    ctx, task = make_ctx_and_task(
        task_cls=SuggestDtypeOptimizations,
        current_df=df,
        task_overrides={"min_savings_bytes": 0},  # critical
        global_overrides={"output_dir": str(tmp_path)},
    )

    # Ensure correct metadata so rule applies
    ctx.set_metadata("semantic_types", {"tiny_int": "continuous"})
    ctx.set_metadata("inferred_dtypes", {"tiny_int": "int64"})

    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"

    suggestions = result.data["suggestions"]

    # This is the key assertion: MUST NOT be filtered out
    assert "tiny_int" in suggestions
    assert suggestions["tiny_int"]["suggested_dtype"] == "int8"
