# tests/eda/test_tasks/test_compute_acf_pacf.py


from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.compute_acf_pacf import ComputeACFPACF
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from collections.abc import Generator

    from pandas import DataFrame, DatetimeIndex


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ts_config(
    index_col: str = "date",
    value_cols: list | None = None,
    frequency: str | None = "D",
    max_lags: int = 20,
    alpha: float = 0.05,
) -> dict:
    """Build a minimal config dict with time series enabled."""
    return {
        "tasks": {
            "time_series": {
                "enabled": True,
                "datetime_index_column": index_col,
                "value_columns": value_cols or [],
                "frequency": frequency,
                "group_by_column": None,
                "tasks": {
                    "acf_pacf": {
                        "max_lags": max_lags,
                        "alpha": alpha,
                    },
                },
            },
        },
    }


def _make_ar1_df(n: int = 100, phi: float = 0.8) -> pd.DataFrame:
    """Generate a strongly autocorrelated AR(1) series."""
    rng: Generator = np.random.default_rng(42)
    values: list[float] = [0.0]
    for _ in range(n - 1):
        values.append(phi * values[-1] + rng.normal(0, 1))
    dates: DatetimeIndex = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "value": values})


def _make_white_noise_df(n: int = 100) -> pd.DataFrame:
    """Generate a white noise series with no autocorrelation."""
    rng: Generator = np.random.default_rng(0)
    dates: DatetimeIndex = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "value": rng.normal(0, 1, n)})


def _run(df, config_override=None, **kwargs) -> TaskResult:
    """Convenience: create ctx+task with TS config and run."""
    config: dict = config_override or _ts_config(**kwargs)
    # max_lags must be passed via task_overrides so get_task_param() finds it
    # under config["tasks"]["compute_acf_pacf"]["max_lags"], not buried inside
    # the time_series nested config which get_task_param does not read.
    task_overrides: dict[str, Any] = {
        k: v for k, v in kwargs.items() if k in ("max_lags", "alpha")
    }
    ctx, task = make_ctx_and_task(
        task_cls=ComputeACFPACF,
        current_df=df,
        task_overrides=task_overrides,
        global_overrides={"output_dir": "/tmp/test_acf"},
    )
    ctx.config = config
    ctx.set_metadata(
        "semantic_types",
        dict.fromkeys(df.select_dtypes(include="number").columns, "continuous")
        | {"date": "datetime"},
    )
    return ctx.run_task(task)


# ── Disabled / misconfigured ──────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore")
def test_disabled_returns_success_with_message() -> None:
    df: DataFrame = _make_ar1_df()
    ctx, task = make_ctx_and_task(
        task_cls=ComputeACFPACF,
        current_df=df,
        global_overrides={"output_dir": "/tmp"},
    )
    ctx.config = {"tasks": {"time_series": {"enabled": False}}}
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.summary["time_series_enabled"] is False


@pytest.mark.filterwarnings("ignore")
def test_missing_index_col_returns_graceful_result() -> None:
    df: DataFrame = _make_ar1_df()
    ctx, task = make_ctx_and_task(
        task_cls=ComputeACFPACF,
        current_df=df,
        global_overrides={"output_dir": "/tmp"},
    )
    ctx.config = {
        "tasks": {
            "time_series": {
                "enabled": True,
                "datetime_index_column": None,
                "value_columns": [],
                "frequency": None,
                "group_by_column": None,
                "tasks": {"acf_pacf": {"max_lags": 10, "alpha": 0.05}},
            },
        },
    }
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.data == {}


@pytest.mark.filterwarnings("ignore")
def test_wrong_index_col_returns_graceful_result() -> None:
    df: DataFrame = _make_ar1_df()
    result: TaskResult = _run(df, index_col="nonexistent_col")
    assert result.status == "success"
    assert result.data == {}


# ── Core ACF/PACF computation ─────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore")
def test_ar1_series_has_significant_acf_lags() -> None:
    """A strongly autocorrelated AR(1) series must have significant ACF lags."""
    df: DataFrame = _make_ar1_df(n=150, phi=0.9)
    result: TaskResult = _run(df, max_lags=20)
    assert result.status == "success"
    assert "value" in result.data
    acf_sig = result.data["value"]["acf_significant_lags"]
    assert len(acf_sig) > 0
    assert 1 in acf_sig  # lag 1 must be significant for strong AR(1)


@pytest.mark.filterwarnings("ignore")
def test_white_noise_has_few_significant_lags() -> None:
    """White noise should have very few (ideally zero) significant lags."""
    df: DataFrame = _make_white_noise_df(n=200)
    result: TaskResult = _run(df, max_lags=20)
    assert result.status == "success"
    if "value" in result.data:
        acf_sig = result.data["value"]["acf_significant_lags"]
        # At α=0.05 expect ~5% false positives → at most ~2 of 20 lags
        assert len(acf_sig) <= 3


@pytest.mark.filterwarnings("ignore")
def test_output_structure_complete() -> None:
    """Each column result must contain all expected keys."""
    df: DataFrame = _make_ar1_df(n=100)
    result: TaskResult = _run(df, max_lags=10)
    assert result.status == "success"
    assert "value" in result.data
    entry = result.data["value"]
    for key in (
        "lags",
        "acf_values",
        "pacf_values",
        "acf_confidence_lower",
        "acf_confidence_upper",
        "pacf_confidence_lower",
        "pacf_confidence_upper",
        "acf_significant_lags",
        "pacf_significant_lags",
        "n",
        "max_lags",
        "alpha",
    ):
        assert key in entry


@pytest.mark.filterwarnings("ignore")
def test_arrays_equal_length() -> None:
    """All lag arrays must be the same length."""
    df: DataFrame = _make_ar1_df(n=100)
    result: TaskResult = _run(df, max_lags=15)
    entry = result.data["value"]
    n_lags: int = len(entry["lags"])
    for key in (
        "acf_values",
        "pacf_values",
        "acf_confidence_lower",
        "acf_confidence_upper",
        "pacf_confidence_lower",
        "pacf_confidence_upper",
    ):
        assert len(entry[key]) == n_lags


@pytest.mark.filterwarnings("ignore")
def test_acf_lag_zero_is_one() -> None:
    """ACF at lag 0 must always be 1.0."""
    df: DataFrame = _make_ar1_df(n=100)
    result: TaskResult = _run(df, max_lags=10)
    assert abs(result.data["value"]["acf_values"][0] - 1.0) < 1e-4


@pytest.mark.filterwarnings("ignore")
def test_max_lags_respected() -> None:
    """Number of lags computed must not exceed max_lags."""
    df: DataFrame = _make_ar1_df(n=200)
    result: TaskResult = _run(df, max_lags=15)
    n_lags: int = len(result.data["value"]["lags"])
    assert n_lags <= 16  # lags 0..max_lags inclusive


@pytest.mark.filterwarnings("ignore")
def test_short_series_reduces_lags() -> None:
    """A short series must reduce max_lags rather than fail."""
    n = 25
    dates: DatetimeIndex = pd.date_range("2020-01-01", periods=n, freq="D")
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame({"date": dates, "value": rng.normal(0, 1, n)})
    result: TaskResult = _run(df, max_lags=40)
    assert result.status == "success"
    if "value" in result.data:
        assert result.data["value"]["max_lags"] < 40


@pytest.mark.filterwarnings("ignore")
def test_multiple_value_columns(tmp_path) -> None:
    """Multiple value columns must each produce a result entry."""
    n = 100
    rng: Generator = np.random.default_rng(1)
    dates: DatetimeIndex = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "col_a": rng.normal(0, 1, n),
            "col_b": rng.normal(0, 1, n),
        }
    )
    result: TaskResult = _run(df, max_lags=10)
    assert result.status == "success"
    assert "col_a" in result.data
    assert "col_b" in result.data


@pytest.mark.filterwarnings("ignore")
def test_guidance_emitted() -> None:
    """EDA guidance must be attached for each computed column."""
    df: DataFrame = _make_ar1_df(n=150, phi=0.9)
    result: TaskResult = _run(df, max_lags=20)
    assert result.status == "success"
    assert result.guidance is not None
    assert "value" in result.guidance
    assert len(result.guidance["value"]["eda"]) > 0


@pytest.mark.filterwarnings("ignore")
def test_summary_counts_correct() -> None:
    """columns_computed must equal number of entries in data."""
    df: DataFrame = _make_ar1_df(n=100)
    result: TaskResult = _run(df, max_lags=10)
    assert result.summary["columns_computed"] == len(result.data)


def test_no_plots_generated() -> None:
    df: DataFrame = _make_ar1_df(n=100)
    result: TaskResult = _run(df, max_lags=10)
    assert result.plots is None
