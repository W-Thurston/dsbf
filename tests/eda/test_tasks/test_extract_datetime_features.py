# tests/eda/test_tasks/test_extract_datetime_features.py


from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.extract_datetime_features import (
    ExtractDatetimeFeatures,
    _relevant_features,
    _temporal_summary,
)
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from pandas import Series, Timestamp

    from dsbf.eda.task_result import TaskResult

# ── Unit tests for pure helper functions ──────────────────────────────────────


def _make_series(start: str, periods: int, freq: str = "D") -> pd.Series:
    return pd.Series(pd.date_range(start=start, periods=periods, freq=freq))


def test_temporal_summary_basic() -> None:
    s: Series = _make_series("2020-01-01", 365)
    summary: dict[str, Any] = _temporal_summary(s)
    assert summary["range_days"] == 364
    assert summary["n_unique_dates"] == 365
    assert "year" in summary["dominant_components"]
    assert "month" in summary["dominant_components"]


def test_temporal_summary_has_time_component() -> None:
    s: Series[Timestamp] = pd.Series(
        pd.date_range("2020-01-01 08:00", periods=100, freq="h")
    )
    summary: dict[str, Any] = _temporal_summary(s)
    assert summary["has_time_component"] is True
    assert "hour" in summary["dominant_components"]


def test_temporal_summary_no_time_component() -> None:
    s: Series = _make_series("2020-01-01", 100, freq="D")
    summary: dict[str, Any] = _temporal_summary(s)
    assert summary["has_time_component"] is False


def test_relevant_features_always_includes_days_since_epoch() -> None:
    # Minimal column - short range, single day
    summary: dict[str, bool | int | list] = {
        "range_days": 10,
        "has_time_component": False,
        "dominant_components": [],
    }
    features: list[str] = _relevant_features(summary)
    assert "days_since_epoch" in features


def test_relevant_features_year_only_for_multi_year_range() -> None:
    summary_short: dict[str, bool | int | list[str]] = {
        "range_days": 100,
        "has_time_component": False,
        "dominant_components": ["month", "day_of_week"],
    }
    summary_long: dict[str, bool | int | list[str]] = {
        "range_days": 800,
        "has_time_component": False,
        "dominant_components": ["year", "month", "day_of_week"],
    }
    assert "year" not in _relevant_features(summary_short)
    assert "year" in _relevant_features(summary_long)


def test_relevant_features_hour_only_when_time_present() -> None:
    summary_no_time: dict[str, bool | int | list[str]] = {
        "range_days": 365,
        "has_time_component": False,
        "dominant_components": ["year", "month", "day_of_week"],
    }
    summary_with_time: dict[str, bool | int | list[str]] = {
        "range_days": 365,
        "has_time_component": True,
        "dominant_components": ["year", "month", "day_of_week", "hour"],
    }
    assert "hour" not in _relevant_features(summary_no_time)
    assert "hour" in _relevant_features(summary_with_time)


def test_relevant_features_cyclic_included_with_month() -> None:
    summary: dict[str, bool | int | list[str]] = {
        "range_days": 400,
        "has_time_component": False,
        "dominant_components": ["year", "month", "day_of_week"],
    }
    features: list[str] = _relevant_features(summary)
    assert "month_sin_cos" in features
    assert "day_of_week_sin_cos" in features


def test_relevant_features_no_duplicates() -> None:
    summary: dict[str, bool | int | list[str]] = {
        "range_days": 800,
        "has_time_component": True,
        "dominant_components": ["year", "month", "day_of_week", "hour"],
    }
    features: list[str] = _relevant_features(summary)
    assert len(features) == len(set(features))


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_datetime_column_analysed(tmp_path) -> None:
    """A datetime-typed column must be analysed and produce recommendations."""
    df = pd.DataFrame(
        {"event_date": pd.date_range("2020-01-01", periods=365, freq="D")},
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ExtractDatetimeFeatures,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"event_date": "datetime"})
    result: TaskResult = run_task_with_dependencies(ctx, ExtractDatetimeFeatures)

    assert result.status == "success"
    assert "event_date" in result.data
    assert "temporal_summary" in result.data["event_date"]
    assert "recommended_features" in result.data["event_date"]
    assert len(result.data["event_date"]["recommended_features"]) > 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_temporal_summary_in_result(tmp_path) -> None:
    """Temporal summary must contain expected keys."""
    df = pd.DataFrame({"ts": pd.date_range("2019-06-01", periods=500, freq="D")})

    ctx, _ = make_ctx_and_task(
        task_cls=ExtractDatetimeFeatures,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"ts": "datetime"})
    result: TaskResult = run_task_with_dependencies(ctx, ExtractDatetimeFeatures)

    assert result.status == "success"
    s = result.data["ts"]["temporal_summary"]
    for key in (
        "min",
        "max",
        "range_days",
        "n_unique_dates",
        "has_time_component",
        "dominant_components",
    ):
        assert key in s


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_hourly_data_recommends_hour_features(tmp_path) -> None:
    """Hourly data must recommend hour and hour_sin_cos features."""
    df = pd.DataFrame({"ts": pd.date_range("2021-01-01", periods=1000, freq="h")})

    ctx, _ = make_ctx_and_task(
        task_cls=ExtractDatetimeFeatures,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"ts": "datetime"})
    result: TaskResult = run_task_with_dependencies(ctx, ExtractDatetimeFeatures)

    assert result.status == "success"
    features = result.data["ts"]["recommended_features"]
    assert "hour" in features
    assert "hour_sin_cos" in features


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_multi_year_data_recommends_year(tmp_path) -> None:
    """Data spanning more than a year must recommend year extraction."""
    df = pd.DataFrame({"date": pd.date_range("2018-01-01", periods=1000, freq="D")})

    ctx, _ = make_ctx_and_task(
        task_cls=ExtractDatetimeFeatures,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"date": "datetime"})
    result: TaskResult = run_task_with_dependencies(ctx, ExtractDatetimeFeatures)

    assert result.status == "success"
    assert "year" in result.data["date"]["recommended_features"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_non_datetime_column_not_analysed(tmp_path) -> None:
    """A column typed as continuous must not appear in datetime results."""
    df = pd.DataFrame(
        {
            "revenue": [1.0, 2.0, 3.0] * 100,
            "date": pd.date_range("2020-01-01", periods=300, freq="D"),
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ExtractDatetimeFeatures,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {
            "revenue": "continuous",
            "date": "datetime",
        },
    )
    result: TaskResult = run_task_with_dependencies(ctx, ExtractDatetimeFeatures)

    assert result.status == "success"
    assert "revenue" not in result.data
    assert "date" in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_attached_eda_and_ml(tmp_path) -> None:
    """Both EDA and ML guidance blurbs must be attached for datetime columns."""
    df = pd.DataFrame(
        {"created_at": pd.date_range("2020-01-01", periods=365, freq="D")}
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ExtractDatetimeFeatures,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"created_at": "datetime"})
    result: TaskResult = run_task_with_dependencies(ctx, ExtractDatetimeFeatures)

    assert result.status == "success"
    assert result.guidance is not None
    assert "created_at" in result.guidance
    assert len(result.guidance["created_at"]["eda"]) > 0
    assert len(result.guidance["created_at"]["ml"]) > 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_ml_guidance_has_extract_feature_actions(tmp_path) -> None:
    """ML guidance must contain extract_feature action chips."""
    df = pd.DataFrame({"event": pd.date_range("2020-01-01", periods=400, freq="D")})

    ctx, _ = make_ctx_and_task(
        task_cls=ExtractDatetimeFeatures,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"event": "datetime"})
    result: TaskResult = run_task_with_dependencies(ctx, ExtractDatetimeFeatures)

    assert result.status == "success"
    ml_actions = result.guidance["event"]["ml"][0]["actions"]
    assert len(ml_actions) > 0
    assert all(a["action"] == "extract_feature" for a in ml_actions)
    # Each action must name the feature and provide an ml_note
    for action in ml_actions:
        assert "feature" in action
        assert "ml_note" in action
        assert "description" in action


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_correct(tmp_path) -> None:
    """Summary feature count must equal sum of recommended features across columns."""
    df = pd.DataFrame(
        {
            "date_a": pd.date_range("2020-01-01", periods=400, freq="D"),
            "date_b": pd.date_range("2018-01-01", periods=400, freq="D"),
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ExtractDatetimeFeatures,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"date_a": "datetime", "date_b": "datetime"})
    result: TaskResult = run_task_with_dependencies(ctx, ExtractDatetimeFeatures)

    assert result.status == "success"
    total: int = sum(len(v["recommended_features"]) for v in result.data.values())
    assert result.summary["total_features_recommended"] == total
    assert result.summary["datetime_columns_found"] == 2


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    dates: list[Timestamp] = pd.date_range("2021-01-01", periods=200, freq="D").tolist()
    df = pl.DataFrame({"event_date": dates})

    ctx, _ = make_ctx_and_task(
        task_cls=ExtractDatetimeFeatures,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"event_date": "datetime"})
    result: TaskResult = run_task_with_dependencies(ctx, ExtractDatetimeFeatures)

    assert result.status == "success"
    assert "event_date" in result.data


def test_no_plots_generated(tmp_path) -> None:
    """Datetime features task must not generate plots."""
    df = pd.DataFrame({"d": pd.date_range("2020-01-01", periods=100, freq="D")})

    ctx, _ = make_ctx_and_task(
        task_cls=ExtractDatetimeFeatures,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"d": "datetime"})
    result: TaskResult = run_task_with_dependencies(ctx, ExtractDatetimeFeatures)

    assert result.status == "success"
    assert result.plots is None
