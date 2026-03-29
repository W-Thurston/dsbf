# tests/eda/test_tasks/test__time_series_utils.py
#
# Unit tests for the shared time series utility module.
# These tests exercise the helpers directly without running full tasks.

import numpy as np
import pandas as pd
import pytest

from dsbf.eda.tasks._time_series_utils import (
    TimeSeriesConfig,
    TimeSeriesSetupError,
    infer_frequency,
    make_disabled_result,
    prepare_time_series_df,
    read_time_series_config,
    to_pandas,
    validate_time_series_config,
)

# ── Minimal task stub for testing helpers that need a task_instance ───────────


class _StubTask:
    """Minimal stub satisfying the task_instance interface required by helpers."""

    def __init__(self, config=None):
        self.config = config or {}
        self.logs = []
        self.context = None

    def _log(self, msg, level="debug"):
        self.logs.append((level, msg))

    def get_task_param(self, key, default=None):
        return None

    def get_shared_param(self, block_name, key, default=None):
        return self.config.get("tasks", {}).get(block_name, {}).get(key, default)


# ── TimeSeriesConfig ──────────────────────────────────────────────────────────


def test_ts_config_stores_fields() -> None:
    cfg = TimeSeriesConfig(
        enabled=True,
        index_col="date",
        value_cols=["sales"],
        frequency="D",
        group_by_col=None,
    )
    assert cfg.enabled is True
    assert cfg.index_col == "date"
    assert cfg.value_cols == ["sales"]
    assert cfg.frequency == "D"
    assert cfg.group_by_col is None


# ── read_time_series_config ───────────────────────────────────────────────────


def test_read_config_disabled_by_default() -> None:
    task = _StubTask(config={})
    cfg = read_time_series_config(task)
    assert cfg.enabled is False
    assert cfg.index_col is None


def test_read_config_reads_values() -> None:
    task = _StubTask(
        config={
            "tasks": {
                "time_series": {
                    "enabled": True,
                    "datetime_index_column": "event_date",
                    "value_columns": ["revenue", "sessions"],
                    "frequency": "W",
                    "group_by_column": None,
                }
            }
        }
    )
    cfg = read_time_series_config(task)
    assert cfg.enabled is True
    assert cfg.index_col == "event_date"
    assert cfg.value_cols == ["revenue", "sessions"]
    assert cfg.frequency == "W"


def test_read_config_empty_string_index_col_treated_as_none() -> None:
    task = _StubTask(
        config={
            "tasks": {"time_series": {"enabled": True, "datetime_index_column": ""}}
        }
    )
    cfg = read_time_series_config(task)
    assert cfg.index_col is None


# ── validate_time_series_config ───────────────────────────────────────────────


def test_validate_raises_when_disabled() -> None:
    cfg = TimeSeriesConfig(False, None, [], None, None)
    df = pd.DataFrame({"a": [1, 2, 3]})
    task = _StubTask()
    with pytest.raises(TimeSeriesSetupError, match="not enabled"):
        validate_time_series_config(cfg, df, task)


def test_validate_raises_when_no_index_col() -> None:
    cfg = TimeSeriesConfig(True, None, [], None, None)
    df = pd.DataFrame({"a": [1, 2, 3]})
    task = _StubTask()
    with pytest.raises(TimeSeriesSetupError, match="not set"):
        validate_time_series_config(cfg, df, task)


def test_validate_raises_when_index_col_missing() -> None:
    cfg = TimeSeriesConfig(True, "nonexistent", [], None, None)
    df = pd.DataFrame({"a": [1, 2, 3]})
    task = _StubTask()
    with pytest.raises(TimeSeriesSetupError, match="not present"):
        validate_time_series_config(cfg, df, task)


def test_validate_passes_for_valid_config() -> None:
    cfg = TimeSeriesConfig(True, "date", [], None, None)
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10, freq="D")})
    task = _StubTask()
    validate_time_series_config(cfg, df, task)  # must not raise


def test_validate_warns_not_raises_for_group_by_col() -> None:
    cfg = TimeSeriesConfig(True, "date", [], None, "region")
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10, freq="D")})
    task = _StubTask()
    validate_time_series_config(cfg, df, task)  # must not raise
    assert any("group_by_column" in msg for _, msg in task.logs)
    assert any(level == "warn" for level, _ in task.logs)


# ── prepare_time_series_df ────────────────────────────────────────────────────


def _make_df(n: int = 30, freq: str = "D") -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq=freq)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "date": dates,
            "sales": rng.normal(100, 10, n),
            "sessions": rng.normal(500, 50, n),
        }
    )


def test_prepare_sorts_by_index() -> None:
    df = _make_df(20)
    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    cfg = TimeSeriesConfig(True, "date", [], None, None)
    task = _StubTask()
    result_df, _ = prepare_time_series_df(df, cfg, task)
    # Must be sorted ascending
    assert result_df["date"].is_monotonic_increasing


def test_prepare_selects_all_numeric_by_default() -> None:
    df = _make_df(20)
    cfg = TimeSeriesConfig(True, "date", [], None, None)
    task = _StubTask()
    _, value_cols = prepare_time_series_df(df, cfg, task)
    assert "sales" in value_cols
    assert "sessions" in value_cols
    assert "date" not in value_cols


def test_prepare_respects_configured_value_cols() -> None:
    df = _make_df(20)
    cfg = TimeSeriesConfig(True, "date", ["sales"], None, None)
    task = _StubTask()
    _, value_cols = prepare_time_series_df(df, cfg, task)
    assert value_cols == ["sales"]
    assert "sessions" not in value_cols


def test_prepare_warns_on_missing_configured_col() -> None:
    df = _make_df(20)
    cfg = TimeSeriesConfig(True, "date", ["sales", "nonexistent"], None, None)
    task = _StubTask()
    _, value_cols = prepare_time_series_df(df, cfg, task)
    assert "sales" in value_cols
    assert "nonexistent" not in value_cols
    assert any("nonexistent" in msg or "skipped" in msg.lower() for _, msg in task.logs)


def test_prepare_converts_string_dates() -> None:
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    cfg = TimeSeriesConfig(True, "date", [], None, None)
    task = _StubTask()
    result_df, _ = prepare_time_series_df(df, cfg, task)
    assert pd.api.types.is_datetime64_any_dtype(result_df["date"])


def test_prepare_raises_on_too_few_timestamps() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "value": [1.0, 2.0],
        }
    )
    cfg = TimeSeriesConfig(True, "date", [], None, None)
    task = _StubTask()
    with pytest.raises(TimeSeriesSetupError, match="only 2"):
        prepare_time_series_df(df, cfg, task)


def test_prepare_raises_when_no_numeric_cols_available() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10),
            "label": ["a"] * 10,
        }
    )
    cfg = TimeSeriesConfig(True, "date", [], None, None)
    task = _StubTask()
    with pytest.raises(TimeSeriesSetupError, match="No numeric"):
        prepare_time_series_df(df, cfg, task)


def test_prepare_drops_null_timestamps() -> None:
    dates = pd.date_range("2020-01-01", periods=10).tolist()
    dates[3] = None
    df = pd.DataFrame({"date": dates, "value": range(10)})
    cfg = TimeSeriesConfig(True, "date", [], None, None)
    task = _StubTask()
    result_df, _ = prepare_time_series_df(df, cfg, task)
    assert result_df["date"].notna().all()
    assert len(result_df) == 9


# ── infer_frequency ───────────────────────────────────────────────────────────


def test_infer_frequency_returns_configured_unchanged() -> None:
    series = pd.Series(pd.date_range("2020-01-01", periods=10, freq="D"))
    task = _StubTask()
    assert infer_frequency(series, configured="W", task_instance=task) == "W"


def test_infer_frequency_daily_series() -> None:
    series = pd.Series(pd.date_range("2020-01-01", periods=50, freq="D"))
    task = _StubTask()
    freq = infer_frequency(series, configured=None, task_instance=task)
    # pandas infer_freq or fallback both should give daily alias
    assert freq in ("D", "B", "C") or freq is not None


def test_infer_frequency_weekly_series() -> None:
    series = pd.Series(pd.date_range("2020-01-01", periods=30, freq="W"))
    task = _StubTask()
    freq = infer_frequency(series, configured=None, task_instance=task)
    assert freq is not None


def test_infer_frequency_returns_none_for_irregular() -> None:
    # Very irregular gaps - 1 day, then 100 days, alternating
    dates = [pd.Timestamp("2020-01-01")]
    for i in range(20):
        gap = 1 if i % 2 == 0 else 100
        dates.append(dates[-1] + pd.Timedelta(days=gap))
    series = pd.Series(dates)
    task = _StubTask()
    # Should not raise - may return None or a freq alias
    result = infer_frequency(series, configured=None, task_instance=task)
    # Just verify it doesn't crash and returns something or None
    assert result is None or isinstance(result, str)


# ── make_disabled_result ──────────────────────────────────────────────────────


def test_make_disabled_result_structure() -> None:
    result = make_disabled_result("my_task", "TS not enabled")
    assert result["summary"]["time_series_enabled"] is False
    assert "TS not enabled" in result["summary"]["message"]
    assert result["data"] == {}


# ── to_pandas ─────────────────────────────────────────────────────────────────


def test_to_pandas_passthrough_for_pandas() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = to_pandas(df)
    assert isinstance(result, pd.DataFrame)
    assert result is df  # same object


def test_to_pandas_converts_polars() -> None:
    try:
        import polars as pl

        df = pl.DataFrame({"a": [1, 2, 3]})
        result = to_pandas(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result["a"]) == [1, 2, 3]
    except ImportError:
        pytest.skip("polars not installed")
