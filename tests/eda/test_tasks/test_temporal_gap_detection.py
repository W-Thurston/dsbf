# tests/eda/test_tasks/test_temporal_gap_detection.py


from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.temporal_gap_detection import TemporalGapDetection, _analyse_gaps
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from pandas import DatetimeIndex, Series, Timestamp

    from dsbf.eda.task_result import TaskResult

# ── Unit tests for _analyse_gaps ──────────────────────────────────────────────


def test_analyse_gaps_regular_daily_series() -> None:
    """A perfectly regular daily series must have dominant_gap_days=1."""
    s: Series[Timestamp] = pd.Series(pd.date_range("2020-01-01", periods=100, freq="D"))
    result: dict[str, Any] | None = _analyse_gaps(s)
    assert result is not None
    assert result["dominant_gap_days"] == 1.0
    assert result["large_gap_count"] == 0


def test_analyse_gaps_detects_large_gap() -> None:
    """A gap substantially larger than the dominant interval must be flagged."""
    dates: list[Timestamp] = list(pd.date_range("2020-01-01", periods=50, freq="D"))
    dates += list(pd.date_range("2020-06-01", periods=50, freq="D"))  # ~3-month gap
    s: Series[Timestamp] = pd.Series(dates)
    result: dict[str, Any] | None = _analyse_gaps(s)
    assert result is not None
    assert result["large_gap_count"] >= 1
    assert result["max_gap_days"] > 30


def test_analyse_gaps_structure_complete() -> None:
    """Result must contain all expected keys."""
    s: Series[Timestamp] = pd.Series(pd.date_range("2021-01-01", periods=30, freq="D"))
    result: dict[str, Any] | None = _analyse_gaps(s)
    assert result is not None
    for key in (
        "n_observations",
        "date_min",
        "date_max",
        "total_range_days",
        "dominant_gap_days",
        "median_gap_days",
        "mean_gap_days",
        "max_gap_days",
        "min_gap_days",
        "large_gap_count",
        "large_gap_threshold_days",
        "large_gap_details",
    ):
        assert key in result


def test_analyse_gaps_too_few_observations() -> None:
    """Fewer than 2 observations must return None."""
    s: Series[Timestamp] = pd.Series([pd.Timestamp("2020-01-01")])
    assert _analyse_gaps(s) is None


def test_analyse_gaps_weekly_series() -> None:
    """A weekly series must have dominant_gap_days=7 with no large gaps."""
    s: Series[Timestamp] = pd.Series(pd.date_range("2020-01-01", periods=52, freq="W"))
    result: dict[str, Any] | None = _analyse_gaps(s)
    assert result is not None
    assert result["dominant_gap_days"] == 7.0
    assert result["large_gap_count"] == 0


def test_analyse_gaps_gap_details_cap_at_10() -> None:
    """large_gap_details must contain at most 10 entries."""
    # Create many gaps: daily for 10 days, then skip a week, repeat
    dates: list = []
    for week in range(20):
        base = pd.Timestamp("2020-01-01") + pd.Timedelta(weeks=week * 2)
        dates.append(base)
        dates.append(base + pd.Timedelta(days=1))
    s: Series = pd.Series(sorted(dates))
    result: dict[str, Any] | None = _analyse_gaps(s)
    assert result is not None
    assert len(result["large_gap_details"]) <= 10


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_daily_series_no_large_gaps(tmp_path) -> None:
    """A complete daily series must have zero large gaps."""
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=365, freq="D")})

    ctx, task = make_ctx_and_task(
        task_cls=TemporalGapDetection,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"date": "datetime"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "date" in result.data
    assert result.data["date"]["large_gap_count"] == 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_series_with_gap_flagged(tmp_path) -> None:
    """A series with a 3-month gap must be flagged."""
    dates_before: DatetimeIndex = pd.date_range("2020-01-01", periods=90, freq="D")
    dates_after: DatetimeIndex = pd.date_range("2020-06-01", periods=90, freq="D")
    df = pd.DataFrame({"event_date": list(dates_before) + list(dates_after)})

    ctx, task = make_ctx_and_task(
        task_cls=TemporalGapDetection,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"event_date": "datetime"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["event_date"]["large_gap_count"] >= 1
    assert result.data["event_date"]["max_gap_days"] > 30


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_column_with_gaps(tmp_path) -> None:
    """EDA guidance must be attached for datetime columns with large gaps."""
    dates_before: DatetimeIndex = pd.date_range("2020-01-01", periods=60, freq="D")
    dates_after: DatetimeIndex = pd.date_range("2020-06-01", periods=60, freq="D")
    df = pd.DataFrame({"ts": list(dates_before) + list(dates_after)})

    ctx, task = make_ctx_and_task(
        task_cls=TemporalGapDetection,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"ts": "datetime"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    assert "ts" in result.guidance
    assert len(result.guidance["ts"]["eda"]) > 0
    actions = result.guidance["ts"]["eda"][0]["actions"]
    assert any(a["action"] == "investigate_gaps" for a in actions)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_no_datetime_columns_returns_empty(tmp_path) -> None:
    """A DataFrame with no datetime columns must return a success with empty data."""
    df = pd.DataFrame({"x": range(50), "y": range(50)})

    ctx, task = make_ctx_and_task(
        task_cls=TemporalGapDetection,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data == {}
    assert result.summary["columns_analysed"] == 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_min_n_threshold_respected(tmp_path) -> None:
    """A column with fewer than min_n values must be skipped."""
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5, freq="D")})

    ctx, task = make_ctx_and_task(
        task_cls=TemporalGapDetection,
        current_df=df,
        task_overrides={"min_n": 10},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"date": "datetime"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "date" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_correct(tmp_path) -> None:
    """columns_with_gaps in summary must equal columns with large_gap_count > 0."""
    dates_before: DatetimeIndex = pd.date_range("2020-01-01", periods=50, freq="D")
    dates_after: DatetimeIndex = pd.date_range("2020-06-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "gappy": list(dates_before) + list(dates_after),
            "clean": pd.date_range("2021-01-01", periods=100, freq="D"),
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=TemporalGapDetection,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"gappy": "datetime", "clean": "datetime"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    expected: int = sum(1 for v in result.data.values() if v["large_gap_count"] > 0)
    assert result.summary["columns_with_gaps"] == expected


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    dates: list[Timestamp] = pd.date_range("2020-01-01", periods=100, freq="D").tolist()
    df = pl.DataFrame({"date": dates})

    ctx, task = make_ctx_and_task(
        task_cls=TemporalGapDetection,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"date": "datetime"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "date" in result.data


def test_no_plots_generated(tmp_path) -> None:
    df = pd.DataFrame({"d": pd.date_range("2020-01-01", periods=50, freq="D")})
    ctx, task = make_ctx_and_task(
        task_cls=TemporalGapDetection,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"d": "datetime"})
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.plots is None
