# tests/eda/test_tasks/test_normality_qq_plots.py
#
# Design principles:
#   - DataFrames use np.round(exponential, 1) or np.round(normal, 1) so
#     infer_types classifies them as "continuous" (unique_ratio ~0.1–0.4).
#   - Semantic types are injected via ctx.set_metadata so infer_types is skipped.
#   - The normality_qq_plots task depends on normality_tests as a dep; we let
#     it run for real (exponential data reliably gives non_normal, normal gives
#     normal). We do not inject mock results — the real task is more robust.
#   - include_normal and n_quantiles tests use ctx.run_task(task) directly
#     since make_ctx_and_task guarantees task.config has the overrides.
#   - Unit tests for _compute_qq_data and _deviation_summary are independent.

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.normality_qq_plots import (
    NormalityQQPlots,
    _compute_qq_data,
    _deviation_summary,
)
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    from pandas import Series


# ── Pure-function unit tests ───────────────────────────────────────────────────


def test_compute_qq_data_basic_structure() -> None:
    s: Series = pd.Series(np.random.default_rng(42).normal(0, 1, 200))
    r: dict[str, Any] = _compute_qq_data(s, n_quantiles=50)
    for k in ("theoretical", "empirical", "reference_line", "fit_mean", "fit_std", "n"):
        assert k in r
    assert r["n"] == 200


def test_compute_qq_data_equal_length_lists() -> None:
    s: Series = pd.Series(np.random.default_rng(0).exponential(1.0, 300))
    r: dict[str, Any] = _compute_qq_data(s, n_quantiles=100)
    assert len(r["theoretical"]) == len(r["empirical"])


def test_compute_qq_data_reference_line_two_points() -> None:
    s: Series = pd.Series(np.arange(1, 101, dtype=float))
    r: dict[str, Any] = _compute_qq_data(s)
    assert len(r["reference_line"]["x"]) == 2 and len(r["reference_line"]["y"]) == 2


def test_compute_qq_data_too_few_values_returns_empty() -> None:
    assert _compute_qq_data(pd.Series([1.0, 2.0, 3.0])) == {}


def test_compute_qq_data_zero_std_returns_empty() -> None:
    assert _compute_qq_data(pd.Series([5.0] * 50)) == {}


def test_compute_qq_data_normal_low_deviation() -> None:
    s: Series = pd.Series(np.random.default_rng(42).normal(0, 1, 5000))
    r: dict[str, Any] = _compute_qq_data(s, n_quantiles=200)
    dev: dict[str, Any] = _deviation_summary(r["theoretical"], r["empirical"])
    assert dev["mean_abs_deviation"] < 0.2


def test_compute_qq_data_skewed_nonzero_deviation() -> None:
    s: Series = pd.Series(np.random.default_rng(42).exponential(1.0, 1000))
    r: dict[str, Any] = _compute_qq_data(s, n_quantiles=100)
    dev: dict[str, Any] = _deviation_summary(r["theoretical"], r["empirical"])
    assert dev["mean_abs_deviation"] > 0.0


def test_deviation_summary_keys() -> None:
    r: dict[str, Any] = _deviation_summary(
        list(np.linspace(-2, 2, 50)), list(np.linspace(-1.5, 2.5, 50))
    )
    for k in (
        "mean_abs_deviation",
        "max_deviation",
        "tail_deviation_lower",
        "tail_deviation_upper",
    ):
        assert k in r


def test_deviation_summary_empty_returns_empty() -> None:
    assert _deviation_summary([], []) == {}


def test_n_quantiles_respected() -> None:
    s: Series = pd.Series(np.random.default_rng(0).normal(0, 1, 500))
    r: dict[str, Any] = _compute_qq_data(s, n_quantiles=30)
    assert r["n_quantiles"] <= 30 and len(r["theoretical"]) <= 30


# ── Integration tests ──────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_non_normal_column_gets_qq_data(tmp_path) -> None:
    """A non-normal column must appear in result data."""
    rng: Generator = np.random.default_rng(42)
    # Round to 1dp: unique_ratio ~0.09 → continuous; exponential → non_normal
    df = pd.DataFrame({"skewed": np.round(rng.exponential(1.0, 500), 1)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"skewed": "continuous"})
    result: TaskResult = run_task_with_dependencies(ctx, NormalityQQPlots)

    assert result.status == "success"
    assert "skewed" in result.data
    assert "theoretical" in result.data["skewed"]
    assert "empirical" in result.data["skewed"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_normal_column_excluded_by_default(tmp_path) -> None:
    """A normal column must be excluded unless include_normal=True."""
    rng: Generator = np.random.default_rng(42)
    # Round to 1dp: ~44 unique in 300 → unique_ratio=0.15 → continuous; N(0,1) → normal
    df = pd.DataFrame({"normal_col": np.round(rng.normal(0, 1, 300), 1)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"normal_col": "continuous"})
    result: TaskResult = run_task_with_dependencies(ctx, NormalityQQPlots)

    assert result.status == "success"
    # Normal column is excluded by default (include_normal=False)
    assert "normal_col" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_include_normal_includes_passing_columns(tmp_path) -> None:
    """include_normal=True must include columns that passed normality tests."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"normal_col": np.round(rng.normal(0, 1, 300), 1)})

    # Use ctx.run_task(task) directly so task_overrides are guaranteed to reach
    # the task via task.config (set by make_ctx_and_task).
    ctx, task = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"include_normal": "true"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"normal_col": "continuous"})
    # Also provide a normality_tests result so the task knows the verdict
    ctx.results["normality_tests"] = TaskResult(
        name="normality_tests",
        status="success",
        summary={"message": "mock"},
        data={
            "normal_col": {
                "overall_verdict": "normal",
                "n": 300,
                "alpha": 0.05,
                "primary_test": {
                    "test": "shapiro_wilk",
                    "statistic": 0.99,
                    "p_value": 0.40,
                },
                "jarque_bera": {
                    "test": "jarque_bera",
                    "statistic": 1.0,
                    "p_value": 0.60,
                },
                "primary_verdict": "normal",
                "jb_verdict": "normal",
            }
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "normal_col" in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_skewness_fallback_includes_skewed_column(tmp_path) -> None:
    """
    When normality_tests result is absent, skewness fallback must
    select skewed columns.
    """
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "skewed": np.round(rng.exponential(1.0, 300), 1),
            "normal": np.round(rng.normal(0, 1, 300), 1),
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"skewed": "continuous", "normal": "continuous"})
    # Run WITHOUT normality_tests dep — skip it by not using run_task_with_dependencies
    # so the skewness fallback is exercised
    _, task = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx2, task2 = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx2.set_metadata(
        "semantic_types", {"skewed": "continuous", "normal": "continuous"}
    )
    # No normality_tests in ctx2.results → fallback fires
    result: TaskResult = ctx2.run_task(task2)

    assert result.status == "success"
    # Skewness fallback must have selected skewed (exp skew ~2) over normal (skew ~0)
    assert "skewed" in result.data
    assert result.summary.get("normality_source") == "skewness_fallback"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_output_structure_complete(tmp_path) -> None:
    """Each column entry must contain all expected keys."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"x": np.round(rng.exponential(1.0, 300), 1)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous"})
    result: TaskResult = run_task_with_dependencies(ctx, NormalityQQPlots)

    assert result.status == "success"
    if "x" in result.data:
        e = result.data["x"]
        for k in (
            "theoretical",
            "empirical",
            "reference_line",
            "fit_mean",
            "fit_std",
            "n",
            "n_quantiles",
            "deviation_summary",
            "normality_verdict",
        ):
            assert k in e
        assert len(e["theoretical"]) == len(e["empirical"])


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_n_quantiles_config_respected(tmp_path) -> None:
    """n_quantiles param must control the length of quantile lists."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"x": np.round(rng.exponential(1.0, 500), 1)})

    ctx, task = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 0.5, "n_quantiles": 30},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous"})
    # Use ctx.run_task so task.config (containing n_quantiles=30) is used directly
    ctx.results["normality_tests"] = TaskResult(
        name="normality_tests",
        status="success",
        summary={"message": "mock"},
        data={
            "x": {
                "overall_verdict": "non_normal",
                "n": 500,
                "alpha": 0.05,
                "primary_test": {
                    "test": "shapiro_wilk",
                    "statistic": 0.85,
                    "p_value": 0.0,
                },
                "jarque_bera": {
                    "test": "jarque_bera",
                    "statistic": 80.0,
                    "p_value": 0.0,
                },
                "primary_verdict": "non_normal",
                "jb_verdict": "non_normal",
            }
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "x" in result.data
    assert result.data["x"]["n_quantiles"] <= 30
    assert len(result.data["x"]["theoretical"]) <= 30


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_attached(tmp_path) -> None:
    """EDA guidance must be attached for each computed column."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"skewed": np.round(rng.exponential(1.0, 300), 1)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"skewed": "continuous"})
    result: TaskResult = run_task_with_dependencies(ctx, NormalityQQPlots)

    assert result.status == "success"
    if "skewed" in result.data:
        assert result.guidance is not None
        assert "skewed" in result.guidance
        assert len(result.guidance["skewed"]["eda"]) > 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    rng: Generator = np.random.default_rng(42)
    df = pl.DataFrame({"x": np.round(rng.exponential(1.0, 200), 1).tolist()})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous"})
    result: TaskResult = run_task_with_dependencies(ctx, NormalityQQPlots)

    assert result.status == "success"


def test_no_plots_generated(tmp_path) -> None:
    """Task must not generate static plot files."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"a": np.round(rng.exponential(1.0, 100), 1)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"a": "continuous"})
    result: TaskResult = run_task_with_dependencies(ctx, NormalityQQPlots)

    assert result.status == "success"
    assert result.plots is None
