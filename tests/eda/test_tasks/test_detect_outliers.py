# tests/eda/test_tasks/test_detect_outliers.py


from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.detect_outliers import (
    DetectOutliers,
    _iqr_outliers,
    _mad_outliers,
    _zscore_outliers,
)
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from collections.abc import Generator

    from pandas import Series

    from dsbf.eda.task_result import TaskResult

# ── Unit tests for pure helpers ───────────────────────────────────────────────


def test_iqr_flags_extreme_value() -> None:
    s: Series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 10 + [100.0])
    result: dict[str, Any] = _iqr_outliers(s, len(s))
    assert result["outlier_count"] >= 1
    assert result["outlier_pct"] > 0
    assert result["lower_fence"] < result["upper_fence"]


def test_iqr_clean_column_no_outliers() -> None:
    s: Series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    result: dict[str, Any] = _iqr_outliers(s, len(s))
    assert result["outlier_count"] == 0


def test_zscore_flags_extreme_value() -> None:
    s: Series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 10 + [100.0])
    result: dict[str, Any] | None = _zscore_outliers(s, threshold=3.0, n_rows=len(s))
    assert result is not None
    assert result["outlier_count"] >= 1


def test_zscore_constant_returns_none() -> None:
    s: Series = pd.Series([5.0] * 50)
    assert _zscore_outliers(s, 3.0, 50) is None


def test_mad_flags_extreme_value() -> None:
    s: Series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 10 + [100.0])
    result: dict[str, Any] | None = _mad_outliers(s, threshold=3.5, n_rows=len(s))
    assert result is not None
    assert result["outlier_count"] >= 1


def test_mad_constant_returns_none() -> None:
    s: Series = pd.Series([5.0] * 50)
    assert _mad_outliers(s, 3.5, 50) is None


def test_mad_robust_to_outlier_inflation() -> None:
    """Non-outlier scores must not inflate when an extreme outlier is added."""
    rng: Generator = np.random.default_rng(42)
    base: Series = pd.Series(rng.normal(0, 1, 100))
    with_outlier = pd.concat([base, pd.Series([1000.0])], ignore_index=True)
    r1: dict[str, Any] | None = _mad_outliers(base, 3.5, len(base))
    r2: dict[str, Any] | None = _mad_outliers(with_outlier, 3.5, len(with_outlier))
    assert r1 is not None
    assert r2 is not None
    # The count of flagged non-outlier values must not substantially increase
    # when the extreme value is added - robustness property of MAD
    assert r2["outlier_count"] - r1["outlier_count"] <= 1


# ── Integration tests - univariate methods ────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_all_method_runs_all_three(tmp_path) -> None:
    """method='all' must run IQR, Z-score, and MAD on each column."""
    # 25 distinct floats * 6 = 150 rows, unique_ratio=26/151≈0.17 → continuous
    base: list[float] = (np.linspace(0, 10, 25) * np.ones((6, 25))).flatten().tolist()
    df = pd.DataFrame({"x": [*base, 9999.0]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"method": "all", "run_isolation_forest": "false"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    assert "x" in result.data
    entry = result.data["x"]
    assert "iqr" in entry
    assert "zscore" in entry
    assert "mad" in entry


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_consensus_flag_set_when_multiple_methods_agree(tmp_path) -> None:
    """Consensus must be True when 2+ methods flag the column."""
    # 25 distinct floats * 6 = 150 rows, unique_ratio=26/151≈0.17 → continuous
    base: list[float] = (np.linspace(0, 10, 25) * np.ones((6, 25))).flatten().tolist()
    df = pd.DataFrame({"x": [*base, 9999.0]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"method": "all", "run_isolation_forest": "false"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    assert result.data["x"]["consensus"] is True
    assert len(result.data["x"]["methods_flagging"]) >= 2


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_clean_column_not_flagged(tmp_path) -> None:
    """A perfectly uniform column must have no outliers and no consensus."""
    # Arithmetic sequence - no tails, no outliers by any method
    df = pd.DataFrame({"clean": [float(i) for i in range(1, 201)]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"run_isolation_forest": "false"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    if "clean" in result.data:
        assert result.data["clean"]["consensus"] is False
        assert len(result.data["clean"]["methods_flagging"]) == 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_method_iqr_only(tmp_path) -> None:
    """method='iqr' must only include iqr in the entry."""
    base: list[float] = (np.linspace(0, 10, 25) * np.ones((5, 25))).flatten().tolist()
    df = pd.DataFrame({"x": [*base, 9999.0]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"method": "iqr"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    assert "iqr" in result.data["x"]
    assert "zscore" not in result.data["x"]
    assert "mad" not in result.data["x"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_method_mad_only(tmp_path) -> None:
    """method='mad' must only include mad in the entry."""
    base: list[float] = (np.linspace(0, 10, 25) * np.ones((5, 25))).flatten().tolist()
    df = pd.DataFrame({"x": [*base, 9999.0]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"method": "mad"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    assert "mad" in result.data["x"]
    assert "iqr" not in result.data["x"]
    assert "zscore" not in result.data["x"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_constant_column_not_in_results(tmp_path) -> None:
    """A constant column must be skipped without error."""
    df = pd.DataFrame(
        {
            "const": [5.0] * 50,
            "vary": list(range(50)),
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"run_isolation_forest": "false"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    # const has MAD=0, zscore std=0 - it should appear but with empty
    # methods_flagging, or be skipped due to min_n/zero variance
    if "const" in result.data:
        assert result.data["const"]["consensus"] is False


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_outlier_column(tmp_path) -> None:
    """EDA and ML guidance must be attached for columns with outliers."""
    base: list[float] = (np.linspace(0, 10, 25) * np.ones((5, 25))).flatten().tolist()
    df = pd.DataFrame({"x": [*base, 9999.0]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"method": "all", "run_isolation_forest": "false"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    assert result.guidance is not None
    assert "x" in result.guidance
    assert len(result.guidance["x"]["eda"]) > 0
    assert len(result.guidance["x"]["ml"]) > 0
    ml_actions = result.guidance["x"]["ml"][0]["actions"]
    assert any(a["action"] == "winsorize" for a in ml_actions)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_correct(tmp_path) -> None:
    """columns_with_outliers summary must reflect columns with outliers."""
    # 20 distinct vals * 3 = 60 rows → ratio=21/61≈0.34 → continuous
    base: list[float] = (np.linspace(0, 10, 20) * np.ones((3, 20))).flatten().tolist()
    # clean: 61 distinct integers → ratio=61/61=1.0 → id. Use repeated range.
    clean: list[float] = (
        np.linspace(0, 10, 20) * np.ones((3, 20))
    ).flatten().tolist() + [5.0]
    df = pd.DataFrame(
        {
            "with_outlier": [*base, 9999.0],
            "clean": clean,
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"run_isolation_forest": "false"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    assert result.summary["columns_analysed"] == len(
        [k for k in result.data if k != "__dataset__"],
    )


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_non_numeric_columns_not_in_results(tmp_path) -> None:
    """Non-numeric columns must not appear in results."""
    df = pd.DataFrame(
        {
            "name": ["alice", "bob"] * 25,
            "value": list(range(50)),
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"run_isolation_forest": "false"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    assert "name" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_empty_dataframe_succeeds(tmp_path) -> None:
    """An empty DataFrame must return success with empty data."""
    df = pd.DataFrame()

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    assert result.data in ({}, {"__dataset__": {}})


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    base: list[float] = (np.linspace(0, 10, 25) * np.ones((4, 25))).flatten().tolist()
    df = pl.DataFrame({"x": [*base, 9999.0]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"run_isolation_forest": "false"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    assert "x" in result.data


# ── Integration tests - Isolation Forest ─────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_isolation_forest_runs_and_produces_dataset_entry(tmp_path) -> None:
    """Isolation Forest must populate the __dataset__ sentinel key."""
    rng: Generator = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "a": [*rng.normal(0, 1, n).tolist(), 10.0, -10.0],
            "b": [*rng.normal(0, 1, n).tolist(), -10.0, 10.0],
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={
            "method": "all",
            "run_isolation_forest": "true",
            "contamination": 0.05,
        },
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    assert "__dataset__" in result.data
    assert "isolation_forest" in result.data["__dataset__"]
    if_entry = result.data["__dataset__"]["isolation_forest"]
    assert "n_flagged" in if_entry
    assert "flagged_row_indices" in if_entry
    assert if_entry["n_features_used"] >= 2


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_isolation_forest_guidance_under_dataset_key(tmp_path) -> None:
    """Isolation Forest guidance must be attached under __dataset__ sentinel."""
    rng: Generator = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n).tolist(),
            "b": rng.normal(0, 1, n).tolist(),
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"method": "isolation_forest", "contamination": 0.05},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    if (
        "__dataset__" in result.data
        and "isolation_forest" in result.data["__dataset__"]
    ):
        assert result.guidance is not None
        assert "__dataset__" in result.guidance


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_run_isolation_forest_false_skips_if(tmp_path) -> None:
    """run_isolation_forest=false must skip IF even when method='all'."""
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 100).tolist(),
            "b": rng.normal(0, 1, 100).tolist(),
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"method": "all", "run_isolation_forest": "false"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)

    assert result.status == "success"
    if "__dataset__" in result.data:
        assert "isolation_forest" not in result.data["__dataset__"]


def test_no_plots_generated(tmp_path) -> None:
    """Detect outliers task must not generate plots."""
    df = pd.DataFrame({"x": list(range(50))})
    ctx, _ = make_ctx_and_task(
        task_cls=DetectOutliers,
        current_df=df,
        task_overrides={"run_isolation_forest": "false"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectOutliers)
    assert result.status == "success"
    assert result.plots is None
