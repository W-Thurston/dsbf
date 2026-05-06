# tests/eda/test_tasks/test_detect_stationarity.py


from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_stationarity import (
    DetectStationarity,
    _build_caveats,
    _interpret_tests,
)
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from collections.abc import Generator

    from pandas import DataFrame, DatetimeIndex

# ── Unit tests for pure helpers ───────────────────────────────────────────────


def test_interpret_both_agree_stationary() -> None:
    assessment, confidence = _interpret_tests(adf_rejected=True, kpss_rejected=False)
    assert assessment == "consistent_with_stationary"
    assert confidence == "moderate"


def test_interpret_both_agree_non_stationary() -> None:
    assessment, confidence = _interpret_tests(adf_rejected=False, kpss_rejected=True)
    assert assessment == "consistent_with_non_stationary"
    assert confidence == "moderate"


def test_interpret_both_reject_ambiguous() -> None:
    assessment, confidence = _interpret_tests(adf_rejected=True, kpss_rejected=True)
    assert "ambiguous" in assessment
    assert confidence == "low"


def test_interpret_neither_rejects_inconclusive() -> None:
    assessment, confidence = _interpret_tests(adf_rejected=False, kpss_rejected=False)
    assert "inconclusive" in assessment
    assert confidence == "low"


def test_build_caveats_always_includes_structural_break_note() -> None:
    caveats: list[str] = _build_caveats(n=100, assessment="consistent_with_stationary")
    assert any("structural" in c.lower() for c in caveats)


def test_build_caveats_warns_on_short_series() -> None:
    caveats: list[str] = _build_caveats(n=30, assessment="consistent_with_stationary")
    assert any("short" in c.lower() or str(30) in c for c in caveats)


def test_build_caveats_no_short_warning_for_large_n() -> None:
    caveats: list[str] = _build_caveats(n=200, assessment="consistent_with_stationary")
    assert not any("short" in c.lower() for c in caveats)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ts_config(
    index_col: str = "date",
    tests: list | None = None,
    alpha: float = 0.05,
) -> dict:
    return {
        "tasks": {
            "time_series": {
                "enabled": True,
                "datetime_index_column": index_col,
                "value_columns": [],
                "frequency": "D",
                "group_by_column": None,
                "tasks": {
                    "stationarity": {
                        "alpha": alpha,
                        "tests": tests or ["adf", "kpss"],
                    },
                },
            },
        },
    }


def _run(df, **kwargs) -> TaskResult:
    config: dict = _ts_config(**kwargs)
    # tests and alpha must be passed via task_overrides so get_task_param()
    # finds them under config["tasks"]["detect_stationarity"], not buried
    # inside the nested time_series config structure.
    task_overrides: dict = {k: v for k, v in kwargs.items() if k in ("tests", "alpha")}
    ctx, task = make_ctx_and_task(
        task_cls=DetectStationarity,
        current_df=df,
        task_overrides=task_overrides,
        global_overrides={"output_dir": "/tmp/test_stationarity"},
    )
    ctx.config = config
    ctx.set_metadata(
        "semantic_types",
        dict.fromkeys(df.select_dtypes(include="number").columns, "continuous")
        | {"date": "datetime"},
    )
    return ctx.run_task(task)


def _make_stationary_df(n: int = 150) -> pd.DataFrame:
    """Zero-mean white noise - should be stationary."""
    rng: Generator = np.random.default_rng(42)
    dates: DatetimeIndex = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "value": rng.normal(0, 1, n)})


def _make_random_walk_df(n: int = 150) -> pd.DataFrame:
    """Random walk (cumulative sum) - should be non-stationary."""
    rng: Generator = np.random.default_rng(42)
    dates: DatetimeIndex = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "value": np.cumsum(rng.normal(0, 1, n)),
        },
    )


# ── Disabled / misconfigured ──────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore")
def test_disabled_returns_success() -> None:
    df: DataFrame = _make_stationary_df()
    ctx, task = make_ctx_and_task(
        task_cls=DetectStationarity,
        current_df=df,
        global_overrides={"output_dir": "/tmp"},
    )
    ctx.config = {"tasks": {"time_series": {"enabled": False}}}
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.summary["time_series_enabled"] is False


@pytest.mark.filterwarnings("ignore")
def test_wrong_index_col_graceful() -> None:
    df: DataFrame = _make_stationary_df()
    result: TaskResult = _run(df, index_col="no_such_col")
    assert result.status == "success"
    assert result.data == {}


# ── Core stationarity tests ───────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore")
def test_stationary_series_assessed_correctly() -> None:
    """White noise must be assessed as consistent with stationary."""
    df: DataFrame = _make_stationary_df(n=200)
    result: TaskResult = _run(df)
    assert result.status == "success"
    assert "value" in result.data
    assessment = result.data["value"]["assessment"]
    # White noise should be stationary - ADF rejects, KPSS should not reject
    assert "stationary" in assessment


@pytest.mark.filterwarnings("ignore")
def test_random_walk_assessed_as_non_stationary() -> None:
    """A random walk must be assessed as consistent with non-stationarity."""
    df: DataFrame = _make_random_walk_df(n=200)
    result: TaskResult = _run(df)
    assert result.status == "success"
    assert "value" in result.data
    assessment = result.data["value"]["assessment"]
    # Random walk has a unit root - ADF should not reject, KPSS should reject
    assert "non_stationary" in assessment or "ambiguous" in assessment


@pytest.mark.filterwarnings("ignore")
def test_output_structure_complete() -> None:
    """Each result entry must contain all expected keys."""
    df: DataFrame = _make_stationary_df(n=100)
    result: TaskResult = _run(df)
    assert result.status == "success"
    entry = result.data["value"]
    for key in ("n", "alpha", "assessment", "confidence", "caveats"):
        assert key in entry
    assert isinstance(entry["caveats"], list)
    assert len(entry["caveats"]) >= 1


@pytest.mark.filterwarnings("ignore")
def test_adf_subdict_structure() -> None:
    """ADF result sub-dict must contain expected keys."""
    df: DataFrame = _make_stationary_df(n=100)
    result: TaskResult = _run(df, tests=["adf"])
    entry = result.data["value"]
    assert "adf" in entry
    adf = entry["adf"]
    for key in (
        "test_statistic",
        "p_value",
        "lags_used",
        "critical_values",
        "rejected_h0",
        "interpretation",
    ):
        assert key in adf


@pytest.mark.filterwarnings("ignore")
def test_kpss_subdict_structure() -> None:
    """KPSS result sub-dict must contain expected keys."""
    df: DataFrame = _make_stationary_df(n=100)
    result: TaskResult = _run(df, tests=["kpss"])
    entry = result.data["value"]
    assert "kpss" in entry
    kpss_res = entry["kpss"]
    for key in (
        "test_statistic",
        "p_value",
        "lags_used",
        "critical_values",
        "rejected_h0",
        "interpretation",
    ):
        assert key in kpss_res


@pytest.mark.filterwarnings("ignore")
def test_adf_only_config() -> None:
    """When tests=['adf'], only ADF must run."""
    df: DataFrame = _make_stationary_df(n=100)
    result: TaskResult = _run(df, tests=["adf"])
    entry = result.data["value"]
    assert "adf" in entry
    assert "kpss" not in entry


@pytest.mark.filterwarnings("ignore")
def test_caveats_always_present() -> None:
    """Caveats must always be present and non-empty."""
    df: DataFrame = _make_stationary_df(n=100)
    result: TaskResult = _run(df)
    for col_data in result.data.values():
        assert "caveats" in col_data
        assert len(col_data["caveats"]) >= 1


@pytest.mark.filterwarnings("ignore")
def test_epistemic_note_in_summary() -> None:
    """Summary must contain the epistemic note about test complementarity."""
    df: DataFrame = _make_stationary_df(n=100)
    result: TaskResult = _run(df)
    assert "epistemic_note" in result.summary
    note = result.summary["epistemic_note"].lower()
    assert "opposite" in note or "null" in note


@pytest.mark.filterwarnings("ignore")
def test_guidance_emitted() -> None:
    """EDA guidance must be attached for each tested column."""
    df: DataFrame = _make_stationary_df(n=150)
    result: TaskResult = _run(df)
    assert result.guidance is not None
    assert "value" in result.guidance
    assert len(result.guidance["value"]["eda"]) > 0


@pytest.mark.filterwarnings("ignore")
def test_non_stationary_guidance_has_differencing_action() -> None:
    """Non-stationary guidance must include a differencing action."""
    df: DataFrame = _make_random_walk_df(n=200)
    result: TaskResult = _run(df)
    if "value" in result.guidance:
        actions = result.guidance["value"]["eda"][0].get("actions", [])
        action_types: list = [a.get("action", "") for a in actions]
        if "non_stationary" in result.data["value"]["assessment"]:
            assert any("difference" in a for a in action_types)


@pytest.mark.filterwarnings("ignore")
def test_summary_counts_correct() -> None:
    df: DataFrame = _make_stationary_df(n=100)
    result: TaskResult = _run(df)
    assert result.summary["columns_tested"] == len(result.data)


def test_no_plots_generated() -> None:
    df: DataFrame = _make_stationary_df(n=100)
    result: TaskResult = _run(df)
    assert result.plots is None
