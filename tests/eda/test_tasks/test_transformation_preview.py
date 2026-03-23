# tests/eda/test_tasks/test_transformation_preview.py

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.transformation_preview import (
    _TRANSFORMS,
    TransformationPreview,
    _apply_yeo_johnson,
    _distribution_stats,
    _skew_reduction_pct,
)
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from collections.abc import Generator

    from pandas import Series

# ── Unit tests for pure helper functions ──────────────────────────────────────


def test_distribution_stats_keys() -> None:
    s: Series = pd.Series(np.random.default_rng(0).normal(0, 1, 100))
    stats: dict[str, Any] = _distribution_stats(s)
    for key in ("mean", "median", "std", "skewness", "min", "max", "p5", "p95"):
        assert key in stats


def test_distribution_stats_empty_returns_empty() -> None:
    s: Series[str] = pd.Series([], dtype=float)
    assert _distribution_stats(s) == {}


def test_skew_reduction_pct_improvement() -> None:
    assert _skew_reduction_pct(2.0, 1.0) == 50.0


def test_skew_reduction_pct_zero_before() -> None:
    assert _skew_reduction_pct(0.0, 1.0) == 0.0


def test_skew_reduction_pct_worsening() -> None:
    # After skew is larger — negative reduction
    result: float = _skew_reduction_pct(1.0, 2.0)
    assert result < 0


def test_apply_yeo_johnson_shape_preserved() -> None:
    s: Series = pd.Series(np.random.default_rng(1).normal(0, 1, 100))
    transformed: Series = _apply_yeo_johnson(s)
    assert len(transformed) == len(s)


def test_apply_yeo_johnson_handles_negative_values() -> None:
    s: Series = pd.Series([-5.0, -1.0, 0.0, 1.0, 5.0, 10.0] * 10)
    # Should not raise even with negatives
    transformed: Series = _apply_yeo_johnson(s)
    assert transformed.notna().all()


def test_log1p_precondition_rejects_negatives() -> None:
    t = next(t for t in _TRANSFORMS if t["name"] == "log1p")
    s_neg: Series = pd.Series([-1.0, 0.0, 1.0])
    s_pos: Series = pd.Series([0.0, 1.0, 2.0])
    assert t["precondition"](s_neg) is False
    assert t["precondition"](s_pos) is True


def test_square_precondition_always_true() -> None:
    t = next(t for t in _TRANSFORMS if t["name"] == "square")
    assert t["precondition"](pd.Series([-5.0, 0.0, 5.0])) is True


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_skewed_column_gets_preview(tmp_path) -> None:
    """A right-skewed column must receive a transformation preview."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"income": rng.exponential(scale=1.0, size=500)})

    ctx, task = make_ctx_and_task(
        task_cls=TransformationPreview,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"income": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "income" in result.data
    preview = result.data["income"]
    assert "before_stats" in preview
    assert "transforms" in preview
    assert preview["original_skewness"] > 1.0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_log1p_applied_to_positive_skewed(tmp_path) -> None:
    """log1p must be applied and must reduce skewness for right-skewed positive data."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.exponential(2.0, 500)})

    ctx, task = make_ctx_and_task(
        task_cls=TransformationPreview,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    transforms = result.data["x"]["transforms"]
    assert "log1p" in transforms
    assert not transforms["log1p"].get("skipped", False)
    assert transforms["log1p"]["skew_reduction_pct"] > 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_log1p_skipped_for_negative_values(tmp_path) -> None:
    """log1p must be skipped when the column contains negative values."""
    rng: Generator = np.random.default_rng(42)
    # Heavy-tailed with negatives — Yeo-Johnson should handle it
    df = pd.DataFrame({"x": rng.standard_t(df=2, size=500)})

    ctx, task = make_ctx_and_task(
        task_cls=TransformationPreview,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    if "x" in result.data:
        transforms = result.data["x"]["transforms"]
        if "log1p" in transforms:
            assert transforms["log1p"].get("skipped", False) is True


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_recommended_flag_set_on_best_transform(tmp_path) -> None:
    """Exactly one transform must be marked as recommended per column."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"y": rng.exponential(1.0, 500)})

    ctx, task = make_ctx_and_task(
        task_cls=TransformationPreview,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"y": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    transforms = result.data["y"]["transforms"]
    recommended_count: int = sum(
        1
        for v in transforms.values()
        if not v.get("skipped", False) and v.get("recommended", False)
    )
    assert recommended_count == 1


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_symmetric_column_not_previewed(tmp_path) -> None:
    """A near-symmetric column (|skew| < threshold) must not be previewed."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"normal": rng.normal(0, 1, 500)})

    ctx, task = make_ctx_and_task(
        task_cls=TransformationPreview,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"normal": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "normal" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_reads_skewness_from_detect_skewness_context(tmp_path) -> None:
    """Task must read skewed columns from detect_skewness result when available."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 200),  # not actually skewed
            "b": rng.exponential(1.0, 200),
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=TransformationPreview,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )

    # Inject mock detect_skewness result — only 'a' listed as skewed
    mock_skewness = TaskResult(
        name="detect_skewness",
        status="success",
        summary={"message": "mock"},
        data={"a": {"skewness": 2.5}},  # only 'a', not 'b'
    )
    ctx.results["detect_skewness"] = mock_skewness
    ctx.set_metadata("semantic_types", {"a": "continuous", "b": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    # Only 'a' was in the mock skewness result
    assert "a" in result.data
    assert "b" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_previewed_column(tmp_path) -> None:
    """EDA guidance must be attached for each previewed column."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"income": rng.exponential(1.0, 300)})

    ctx, task = make_ctx_and_task(
        task_cls=TransformationPreview,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"income": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    assert "income" in result.guidance
    assert len(result.guidance["income"]["eda"]) > 0
    action = result.guidance["income"]["eda"][0]["actions"][0]
    assert action["action"] == "apply_transform"
    assert "transform" in action


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_count_correct(tmp_path) -> None:
    """previewed_count in summary must equal number of entries in data."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.exponential(1.0, 300),
            "b": rng.exponential(2.0, 300),
            "c": rng.normal(0, 1, 300),  # symmetric
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=TransformationPreview,
        current_df=df,
        task_overrides={"skew_threshold": 1.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types", {"a": "continuous", "b": "continuous", "c": "continuous"}
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["previewed_count"] == len(result.data)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    rng: Generator = np.random.default_rng(42)
    df = pl.DataFrame({"x": rng.exponential(1.0, 300).tolist()})

    ctx, task = make_ctx_and_task(
        task_cls=TransformationPreview,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "x" in result.data


def test_no_plots_generated(tmp_path) -> None:
    """Transformation preview task must not generate plots."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"a": rng.exponential(1.0, 100)})

    ctx, task = make_ctx_and_task(
        task_cls=TransformationPreview,
        current_df=df,
        task_overrides={"skew_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"a": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
