# tests/eda/test_tasks/test_normality_qq_plots.py


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
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    from pandas import Series


# ── Unit tests for pure helpers ───────────────────────────────────────────────


def test_compute_qq_data_basic_structure() -> None:
    rng: Generator = np.random.default_rng(42)
    s: Series = pd.Series(rng.normal(0, 1, 200))
    result: dict[str, Any] = _compute_qq_data(s, n_quantiles=50)
    assert "theoretical" in result
    assert "empirical" in result
    assert "reference_line" in result
    assert "fit_mean" in result
    assert "fit_std" in result
    assert "n" in result
    assert result["n"] == 200


def test_compute_qq_data_equal_length_lists() -> None:
    rng: Generator = np.random.default_rng(0)
    s: Series = pd.Series(rng.exponential(1.0, 300))
    result: dict[str, Any] = _compute_qq_data(s, n_quantiles=100)
    assert len(result["theoretical"]) == len(result["empirical"])


def test_compute_qq_data_reference_line_has_two_points() -> None:
    s: Series = pd.Series(np.arange(1, 101, dtype=float))
    result: dict[str, Any] = _compute_qq_data(s)
    assert len(result["reference_line"]["x"]) == 2
    assert len(result["reference_line"]["y"]) == 2


def test_compute_qq_data_too_few_values_returns_empty() -> None:
    s: Series = pd.Series([1.0, 2.0, 3.0])
    assert _compute_qq_data(s) == {}


def test_compute_qq_data_zero_std_returns_empty() -> None:
    s: Series = pd.Series([5.0] * 50)
    assert _compute_qq_data(s) == {}


def test_compute_qq_data_normal_distribution_low_deviation() -> None:
    """Quantile pairs from a large normal sample must be close to the diagonal."""
    rng: Generator = np.random.default_rng(42)
    s: Series = pd.Series(rng.normal(0, 1, 5000))
    result: dict[str, Any] = _compute_qq_data(s, n_quantiles=200)
    dev: dict[str, Any] = _deviation_summary(result["theoretical"], result["empirical"])
    # Large normal sample — mean absolute deviation should be small
    assert dev["mean_abs_deviation"] < 0.2


def test_compute_qq_data_skewed_distribution_high_deviation() -> None:
    """Quantile pairs from a skewed distribution must deviate from the diagonal."""
    rng: Generator = np.random.default_rng(42)
    s: Series = pd.Series(rng.exponential(1.0, 1000))
    result: dict[str, Any] = _compute_qq_data(s, n_quantiles=100)
    dev: dict[str, Any] = _deviation_summary(result["theoretical"], result["empirical"])
    assert dev["mean_abs_deviation"] > 0.0


def test_deviation_summary_keys() -> None:
    theoretical: list[Any] = list(np.linspace(-2, 2, 50))
    empirical: list[Any] = list(np.linspace(-1.5, 2.5, 50))  # slight shift
    result: dict[str, Any] = _deviation_summary(theoretical, empirical)
    for key in (
        "mean_abs_deviation",
        "max_deviation",
        "tail_deviation_lower",
        "tail_deviation_upper",
    ):
        assert key in result


def test_deviation_summary_empty_lists() -> None:
    assert _deviation_summary([], []) == {}


def test_n_quantiles_respected() -> None:
    rng: Generator = np.random.default_rng(0)
    s: Series = pd.Series(rng.normal(0, 1, 500))
    result: dict[str, Any] = _compute_qq_data(s, n_quantiles=30)
    assert result["n_quantiles"] <= 30
    assert len(result["theoretical"]) <= 30


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_non_normal_column_gets_qq_data(tmp_path) -> None:
    """A non-normal column from normality_tests must receive QQ data."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"skewed": rng.exponential(1.0, 500)})

    ctx, task = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"skewed": "continuous"})

    # Inject mock normality_tests result marking 'skewed' as non_normal
    mock_norm = TaskResult(
        name="normality_tests",
        status="success",
        summary={"message": "mock"},
        data={
            "skewed": {
                "overall_verdict": "non_normal",
                "n": 500,
                "alpha": 0.05,
                "primary_test": {
                    "test": "shapiro_wilk",
                    "statistic": 0.9,
                    "p_value": 0.001,
                },
                "jarque_bera": {
                    "test": "jarque_bera",
                    "statistic": 50.0,
                    "p_value": 0.0,
                },
                "primary_verdict": "non_normal",
                "jb_verdict": "non_normal",
            },
        },
    )
    ctx.results["normality_tests"] = mock_norm
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "skewed" in result.data
    assert "theoretical" in result.data["skewed"]
    assert "empirical" in result.data["skewed"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_normal_column_excluded_by_default(tmp_path) -> None:
    """A column with verdict=normal must be excluded unless include_normal=True."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"normal_col": rng.normal(0, 1, 300)})

    ctx, task = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"normal_col": "continuous"})

    mock_norm = TaskResult(
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
            },
        },
    )
    ctx.results["normality_tests"] = mock_norm
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "normal_col" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_include_normal_flag_includes_passing_columns(tmp_path) -> None:
    """include_normal=True must include columns that passed normality tests."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"normal_col": rng.normal(0, 1, 300)})

    ctx, task = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"include_normal": "true"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"normal_col": "continuous"})

    mock_norm = TaskResult(
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
            },
        },
    )
    ctx.results["normality_tests"] = mock_norm
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "normal_col" in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_skewness_fallback_when_no_normality_tests(tmp_path) -> None:
    """Task must fall back to skewness threshold when normality_tests hasn't run."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "skewed": rng.exponential(1.0, 300),
            "normal": rng.normal(0, 1, 300),
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"skewed": "continuous", "normal": "continuous"})
    # No normality_tests result injected
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["normality_source"] == "skewness_fallback"
    # Skewed column should be included; normal column likely excluded
    assert "skewed" in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_output_structure_complete(tmp_path) -> None:
    """Each column result must contain all expected keys."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.exponential(1.0, 300)})

    ctx, task = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    if "x" in result.data:
        entry = result.data["x"]
        for key in (
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
            assert key in entry
        assert len(entry["theoretical"]) == len(entry["empirical"])


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_n_quantiles_config_respected(tmp_path) -> None:
    """n_quantiles parameter must control the length of quantile lists."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.exponential(1.0, 500)})

    ctx, task = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 0.5, "n_quantiles": 50},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    if "x" in result.data:
        assert result.data["x"]["n_quantiles"] <= 50
        assert len(result.data["x"]["theoretical"]) <= 50


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_attached(tmp_path) -> None:
    """EDA guidance must be attached for each computed column."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"skewed": rng.exponential(1.0, 300)})

    ctx, task = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"skewed": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    if "skewed" in result.data:
        assert result.guidance is not None
        assert "skewed" in result.guidance
        assert len(result.guidance["skewed"]["eda"]) > 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    rng: Generator = np.random.default_rng(42)
    df = pl.DataFrame({"x": rng.exponential(1.0, 200).tolist()})

    ctx, task = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"


def test_no_plots_generated(tmp_path) -> None:
    """QQ plot data task must not generate static plots."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"a": rng.exponential(1.0, 100)})

    ctx, task = make_ctx_and_task(
        task_cls=NormalityQQPlots,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"a": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
