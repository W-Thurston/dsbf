# tests/eda/test_tasks/test_decompose_time_series.py

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.decompose_time_series import (
    DecomposeTimeSeries,
    _infer_seasonal_period,
)
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy import dtype, float64, floating, ndarray, signedinteger
    from pandas import DataFrame, DatetimeIndex

# ── Unit tests for _infer_seasonal_period ────────────────────────────────────


class _StubTask:
    def __init__(self) -> None:
        self.logs = []

    def _log(self, msg, level="debug") -> None:
        self.logs.append((level, msg))


def test_infer_period_daily_gives_7() -> None:
    task = _StubTask()
    assert _infer_seasonal_period("D", 100, task) == 7


def test_infer_period_monthly_gives_12() -> None:
    task = _StubTask()
    assert _infer_seasonal_period("ME", 100, task) == 12


def test_infer_period_none_frequency_returns_none() -> None:
    task = _StubTask()
    result: int | None = _infer_seasonal_period(None, 100, task)
    assert result is None
    assert any(level == "warn" for level, _ in task.logs)


def test_infer_period_too_short_returns_none() -> None:
    """n < 2 x period must return None."""  # noqa: D403
    task = _StubTask()
    # daily → period=7, need n >= 14; n=10 is too short
    result: int | None = _infer_seasonal_period("D", 10, task)
    assert result is None


def test_infer_period_annual_gives_none() -> None:
    """Annual frequency has period=1 which is invalid for STL."""
    task = _StubTask()
    result: int | None = _infer_seasonal_period("YE", 100, task)
    assert result is None


def test_infer_period_unknown_alias_returns_none() -> None:
    task = _StubTask()
    result: int | None = _infer_seasonal_period("XYZ", 100, task)
    assert result is None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ts_config(
    index_col: str = "date",
    seasonal_period: int | None = 7,
    robust: bool = True,
    frequency: str | None = "D",
) -> dict:
    return {
        "tasks": {
            "time_series": {
                "enabled": True,
                "datetime_index_column": index_col,
                "value_columns": [],
                "frequency": frequency,
                "group_by_column": None,
                "tasks": {
                    "stl_decomposition": {
                        "seasonal_period": seasonal_period,
                        "robust": robust,
                    },
                },
            },
        },
    }


def _run(df, **kwargs) -> TaskResult:
    config: dict = _ts_config(**kwargs)
    # robust and seasonal_period must be passed via task_overrides so
    # get_task_param() finds them under config["tasks"]["decompose_time_series"],
    # not buried inside the nested time_series config structure.
    task_overrides: dict = {
        k: v for k, v in kwargs.items() if k in ("robust", "seasonal_period")
    }
    ctx, task = make_ctx_and_task(
        task_cls=DecomposeTimeSeries,
        current_df=df,
        task_overrides=task_overrides,
        global_overrides={"output_dir": "/tmp/test_stl"},
    )
    ctx.config = config
    ctx.set_metadata(
        "semantic_types",
        dict.fromkeys(df.select_dtypes(include="number").columns, "continuous")
        | {"date": "datetime"},
    )
    return ctx.run_task(task)


def _make_seasonal_df(n: int = 200, period: int = 7) -> pd.DataFrame:
    """Generate a series with a clear seasonal pattern."""
    rng: Generator = np.random.default_rng(42)
    t: ndarray[tuple[int], dtype[signedinteger]] = np.arange(n)
    seasonal: ndarray[tuple[Any, ...], dtype[float64]] = 5.0 * np.sin(
        2 * np.pi * t / period,
    )
    trend: ndarray[tuple[Any, ...], dtype[floating]] = 0.05 * t
    noise = rng.normal(0, 0.5, n)
    dates: DatetimeIndex = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "value": trend + seasonal + noise})


def _make_trend_only_df(n: int = 150) -> pd.DataFrame:
    """Generate a series with a strong linear trend, no seasonality."""
    rng: Generator = np.random.default_rng(1)
    t: ndarray[tuple[int], dtype[signedinteger]] = np.arange(n)
    dates: DatetimeIndex = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "value": 2.0 * t + rng.normal(0, 1, n)})


# ── Disabled / misconfigured ──────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore")
def test_disabled_returns_success() -> None:
    df: DataFrame = _make_seasonal_df()
    ctx, task = make_ctx_and_task(
        task_cls=DecomposeTimeSeries,
        current_df=df,
        global_overrides={"output_dir": "/tmp"},
    )
    ctx.config = {"tasks": {"time_series": {"enabled": False}}}
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.summary["time_series_enabled"] is False


@pytest.mark.filterwarnings("ignore")
def test_wrong_index_col_graceful() -> None:
    df: DataFrame = _make_seasonal_df()
    result: TaskResult = _run(df, index_col="nonexistent")
    assert result.status == "success"
    assert result.data == {}


@pytest.mark.filterwarnings("ignore")
def test_no_seasonal_period_and_no_frequency_graceful() -> None:
    """
    Without period or frequency, task must not raise.

    It either decomposes using an inferred period or
    returns empty data gracefully.
    """
    df: DataFrame = _make_seasonal_df(n=100)
    result: TaskResult = _run(df, seasonal_period=None, frequency=None)
    assert result.status == "success"
    # Either decomposition ran (inferred period) or returned empty - both valid
    assert isinstance(result.data, dict)


# ── Core decomposition ────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore")
def test_seasonal_series_decomposes() -> None:
    """A clearly seasonal series must produce a result."""
    df: DataFrame = _make_seasonal_df(n=200, period=7)
    result: TaskResult = _run(df, seasonal_period=7)
    assert result.status == "success"
    assert "value" in result.data


@pytest.mark.filterwarnings("ignore")
def test_output_arrays_equal_length() -> None:
    """All decomposition arrays must be the same length."""
    df: DataFrame = _make_seasonal_df(n=200)
    result: TaskResult = _run(df, seasonal_period=7)
    entry = result.data["value"]
    n: int = len(entry["timestamps"])
    for key in ("observed", "trend", "seasonal", "residual"):
        assert len(entry[key]) == n, f"{key} length mismatch"


@pytest.mark.filterwarnings("ignore")
def test_output_structure_complete() -> None:
    """Each column result must contain all expected keys."""
    df: DataFrame = _make_seasonal_df(n=200)
    result: TaskResult = _run(df, seasonal_period=7)
    entry = result.data["value"]
    for key in (
        "timestamps",
        "observed",
        "trend",
        "seasonal",
        "residual",
        "trend_strength",
        "seasonal_strength",
        "seasonal_period",
        "robust",
        "n",
    ):
        assert key in entry


@pytest.mark.filterwarnings("ignore")
def test_seasonal_strength_high_for_seasonal_series() -> None:
    """A clearly seasonal series must have seasonal_strength > 0.5."""
    df: DataFrame = _make_seasonal_df(n=300, period=7)
    result: TaskResult = _run(df, seasonal_period=7)
    assert result.data["value"]["seasonal_strength"] > 0.5


@pytest.mark.filterwarnings("ignore")
def test_trend_strength_high_for_trending_series() -> None:
    """A strongly trending series must have trend_strength > 0.5."""
    df: DataFrame = _make_trend_only_df(n=200)
    result: TaskResult = _run(df, seasonal_period=7)
    if "value" in result.data:
        assert result.data["value"]["trend_strength"] > 0.5


@pytest.mark.filterwarnings("ignore")
def test_strength_metrics_in_zero_one_range() -> None:
    """Trend and seasonal strength must be in [0, 1]."""
    df: DataFrame = _make_seasonal_df(n=200)
    result: TaskResult = _run(df, seasonal_period=7)
    entry = result.data["value"]
    assert 0.0 <= entry["trend_strength"] <= 1.0
    assert 0.0 <= entry["seasonal_strength"] <= 1.0


@pytest.mark.filterwarnings("ignore")
def test_seasonal_period_stored_in_result() -> None:
    """The seasonal period used must be stored in the result."""
    df: DataFrame = _make_seasonal_df(n=200)
    result: TaskResult = _run(df, seasonal_period=7)
    assert result.data["value"]["seasonal_period"] == 7
    assert result.summary["seasonal_period"] == 7
    assert result.metadata["seasonal_period"] == 7


@pytest.mark.filterwarnings("ignore")
def test_too_short_series_skipped() -> None:
    """A series shorter than 2xperiod must be skipped without error."""
    n = 10
    dates: DatetimeIndex = pd.date_range("2020-01-01", periods=n, freq="D")
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame({"date": dates, "value": rng.normal(0, 1, n)})
    result: TaskResult = _run(df, seasonal_period=7)
    assert result.status == "success"
    # n=10 < 2x7=14 → column skipped
    assert "value" not in result.data


@pytest.mark.filterwarnings("ignore")
def test_timestamps_are_strings() -> None:
    """Timestamps must be serialisable ISO strings."""
    df: DataFrame = _make_seasonal_df(n=200)
    result: TaskResult = _run(df, seasonal_period=7)
    timestamps = result.data["value"]["timestamps"]
    assert all(isinstance(t, str) for t in timestamps)
    # Verify they look like ISO dates
    assert "T" in timestamps[0] or "-" in timestamps[0]


@pytest.mark.filterwarnings("ignore")
def test_guidance_emitted() -> None:
    """EDA guidance must be attached for each decomposed column."""
    df: DataFrame = _make_seasonal_df(n=200)
    result: TaskResult = _run(df, seasonal_period=7)
    assert result.guidance is not None
    assert "value" in result.guidance
    assert len(result.guidance["value"]["eda"]) > 0


@pytest.mark.filterwarnings("ignore")
def test_guidance_actions_include_seasonal_difference() -> None:
    """Guidance actions must include seasonal_difference."""
    df: DataFrame = _make_seasonal_df(n=300, period=7)
    result: TaskResult = _run(df, seasonal_period=7)
    if "value" in result.guidance:
        actions = result.guidance["value"]["eda"][0].get("actions", [])
        action_types: list = [a.get("action", "") for a in actions]
        assert "seasonal_difference" in action_types


@pytest.mark.filterwarnings("ignore")
def test_robust_flag_stored() -> None:
    df: DataFrame = _make_seasonal_df(n=200)
    result: TaskResult = _run(df, seasonal_period=7, robust=False)
    assert result.data["value"]["robust"] is False
    assert result.metadata["robust"] is False


@pytest.mark.filterwarnings("ignore")
def test_summary_counts_correct() -> None:
    df: DataFrame = _make_seasonal_df(n=200)
    result: TaskResult = _run(df, seasonal_period=7)
    assert result.summary["columns_decomposed"] == len(result.data)


def test_no_plots_generated() -> None:
    df: DataFrame = _make_seasonal_df(n=200)
    result: TaskResult = _run(df, seasonal_period=7)
    assert result.plots is None
