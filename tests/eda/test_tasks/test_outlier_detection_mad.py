# tests/eda/test_tasks/test_outlier_detection_mad.py


from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.outlier_detection_mad import OutlierDetectionMAD, _mad_score
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from collections.abc import Generator

    from pandas import Series

    from dsbf.eda.task_result import TaskResult

# ── Unit tests for _mad_score ─────────────────────────────────────────────────


def test_mad_score_symmetric_outlier() -> None:
    """The extreme value must receive a high Modified Z-score."""
    s: Series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 10 + [100.0])
    scores: Series | None = _mad_score(s)
    assert scores is not None
    assert float(scores.iloc[-1]) > 3.5


def test_mad_score_constant_series_returns_none() -> None:
    """MAD is zero for a constant series - must return None."""
    s: Series = pd.Series([5.0] * 50)
    assert _mad_score(s) is None


def test_mad_score_all_non_negative() -> None:
    """Modified Z-scores must be non-negative (absolute values)."""
    rng: Generator = np.random.default_rng(0)
    s: Series = pd.Series(rng.normal(0, 1, 200))
    scores: Series | None = _mad_score(s)
    assert scores is not None
    assert (scores >= 0).all()


def test_mad_score_symmetry() -> None:
    """A value equidistant below the median must score the same as one above."""
    median = 10.0
    s: Series = pd.Series([median - 5, median, median + 5] * 20)
    scores: Series | None = _mad_score(s)
    assert scores is not None
    low_score = float(scores.iloc[0])
    high_score = float(scores.iloc[2])
    assert abs(low_score - high_score) < 1e-6


def test_mad_score_robust_to_outlier_inflation() -> None:
    """Adding an extreme outlier must not inflate all other scores significantly."""
    rng: Generator = np.random.default_rng(42)
    base: Series = pd.Series(rng.normal(0, 1, 100))
    with_outlier: Series = pd.concat([base, pd.Series([1000.0])], ignore_index=True)

    scores_base: Series | None = _mad_score(base)
    scores_with: Series | None = _mad_score(with_outlier)

    assert scores_base is not None
    assert scores_with is not None
    # Median of base scores should not be inflated by the extreme outlier
    assert abs(float(scores_base.median()) - float(scores_with[:100].median())) < 0.5


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_extreme_outlier_flagged(tmp_path) -> None:
    """An obviously extreme value must be flagged as an outlier."""
    base: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0] * 30
    df = pd.DataFrame({"x": [*base, 10000.0]})

    ctx, task = make_ctx_and_task(
        task_cls=OutlierDetectionMAD,
        current_df=df,
        task_overrides={"threshold": 3.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "x" in result.data
    assert result.data["x"]["outlier_count"] >= 1
    assert 10_000.0 in result.data["x"]["top_outlier_values"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_clean_column_has_no_outliers(tmp_path) -> None:
    """A tightly distributed column must have zero outliers."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"clean": rng.normal(0, 0.1, 200)})

    ctx, _ = make_ctx_and_task(
        task_cls=OutlierDetectionMAD,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, OutlierDetectionMAD)

    assert result.status == "success"
    assert result.data["clean"]["outlier_count"] == 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_threshold_controls_flagging(tmp_path) -> None:
    """A stricter threshold must flag fewer outliers than a lenient one."""
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 300)})

    ctx_strict, task_strict = make_ctx_and_task(
        task_cls=OutlierDetectionMAD,
        current_df=df,
        task_overrides={"threshold": 10.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result_strict: TaskResult = ctx_strict.run_task(task_strict)

    ctx_lenient, task_lenient = make_ctx_and_task(
        task_cls=OutlierDetectionMAD,
        current_df=df,
        task_overrides={"threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result_lenient: TaskResult = ctx_lenient.run_task(task_lenient)

    assert result_strict.status == "success"
    assert result_lenient.status == "success"
    assert (
        result_strict.data["x"]["outlier_count"]
        <= result_lenient.data["x"]["outlier_count"]
    )


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_result_structure_complete(tmp_path) -> None:
    """Each column result must contain all expected keys."""
    base: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
    df = pd.DataFrame({"x": [*base, 999.0]})

    ctx, task = make_ctx_and_task(
        task_cls=OutlierDetectionMAD,
        current_df=df,
        task_overrides={"threshold": 3.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    entry = result.data["x"]
    for key in (
        "outlier_count",
        "outlier_pct",
        "threshold",
        "median",
        "mad",
        "max_modified_z_score",
        "top_outlier_values",
        "n",
    ):
        assert key in entry


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_constant_column_skipped(tmp_path) -> None:
    """A constant column (MAD=0) must be skipped without error."""
    df = pd.DataFrame({"const": [5.0] * 50, "vary": list(range(50))})

    ctx, _ = make_ctx_and_task(
        task_cls=OutlierDetectionMAD,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, OutlierDetectionMAD)

    assert result.status == "success"
    assert "const" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_column_with_outliers(tmp_path) -> None:
    """EDA and ML guidance must be attached for columns with outliers."""
    base: list[float] = [1.0, 2.0, 3.0] * 40
    df = pd.DataFrame({"x": [*base, 9999.0]})

    ctx, task = make_ctx_and_task(
        task_cls=OutlierDetectionMAD,
        current_df=df,
        task_overrides={"threshold": 3.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    assert "x" in result.guidance
    assert len(result.guidance["x"]["eda"]) > 0
    assert len(result.guidance["x"]["ml"]) > 0
    ml_actions = result.guidance["x"]["ml"][0]["actions"]
    assert any(a["action"] == "winsorise" for a in ml_actions)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_min_n_threshold_respected(tmp_path) -> None:
    """Columns with fewer than min_n values must be skipped."""
    df = pd.DataFrame({"tiny": [1.0, 2.0, 100.0] + [None] * 97})

    ctx, task = make_ctx_and_task(
        task_cls=OutlierDetectionMAD,
        current_df=df,
        task_overrides={"min_n": 10},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "tiny" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_correct(tmp_path) -> None:
    """columns_with_outliers summary must equal columns with outlier_count > 0."""
    base: list[float] = [1.0, 2.0, 3.0] * 20
    df = pd.DataFrame(
        {
            "with_outlier": [*base, 9999.0],
            "clean": list(range(61)),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=OutlierDetectionMAD,
        current_df=df,
        task_overrides={"threshold": 3.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    expected: int = sum(1 for v in result.data.values() if v["outlier_count"] > 0)
    assert result.summary["columns_with_outliers"] == expected


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    base: list[float] = [1.0, 2.0, 3.0] * 30
    df = pl.DataFrame({"x": [*base, 9999.0]})

    ctx, task = make_ctx_and_task(
        task_cls=OutlierDetectionMAD,
        current_df=df,
        task_overrides={"threshold": 3.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "x" in result.data


def test_no_plots_generated(tmp_path) -> None:
    df = pd.DataFrame({"a": list(range(50))})
    ctx, _ = make_ctx_and_task(
        task_cls=OutlierDetectionMAD,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, OutlierDetectionMAD)
    assert result.status == "success"
    assert result.plots is None
