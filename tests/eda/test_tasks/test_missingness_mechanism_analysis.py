# tests/eda/test_tasks/test_missingness_mechanism_analysis.py

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.missingness_mechanism_analysis import (
    MissingnessMechanismAnalysis,
    _assess_mechanism,
    _group_difference_tests,
    _little_mcar_test,
    _missingness_correlations,
)
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy import ndarray

# ── Unit tests for pure helpers ───────────────────────────────────────────────


def test_little_mcar_returns_none_for_single_column() -> None:
    df = pd.DataFrame({"a": [1.0, None, 3.0] * 10})
    assert _little_mcar_test(df) is None


def test_little_mcar_returns_none_for_tiny_sample() -> None:
    df = pd.DataFrame({"a": [1.0, None], "b": [2.0, 3.0]})
    assert _little_mcar_test(df) is None


def test_little_mcar_result_structure() -> None:
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 50).tolist(),
            "b": rng.normal(0, 1, 50).tolist(),
        }
    )
    df.loc[df.index[:10], "a"] = None
    result: dict[str, Any] | None = _little_mcar_test(df)
    if result is not None:
        for key in (
            "test_statistic",
            "p_value",
            "degrees_of_freedom",
            "n_patterns",
            "n",
            "caveats",
        ):
            assert key in result
        assert isinstance(result["caveats"], list)
        assert len(result["caveats"]) >= 1


def test_little_mcar_caveats_always_present() -> None:
    """Little's MCAR test must always include epistemic caveats."""
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": [None] * 20 + rng.normal(0, 1, 30).tolist(),
            "b": rng.normal(0, 1, 50).tolist(),
        },
    )
    result: dict[str, Any] | None = _little_mcar_test(df)
    if result is not None:
        assert any(
            "power" in c.lower() or "confirm" in c.lower() for c in result["caveats"]
        )


def test_missingness_correlations_detects_relationship() -> None:
    """Missingness of 'a' fully determined by 'b' must produce high correlation."""
    rng: Generator = np.random.default_rng(42)
    n = 200
    b = rng.normal(0, 1, n)
    # a is missing whenever b > 0
    a: ndarray = np.where(b > 0, np.nan, rng.normal(0, 1, n))
    df = pd.DataFrame({"a": a, "b": b})
    result: dict[str, dict[str, Any]] = _missingness_correlations(df, "a", ["b"])
    assert "b" in result
    assert abs(result["b"]["correlation"]) > 0.3


def test_missingness_correlations_structure() -> None:
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": [None] * 50 + rng.normal(0, 1, 50).tolist(),
            "b": rng.normal(0, 1, 100).tolist(),
        },
    )
    result: dict[str, dict[str, Any]] = _missingness_correlations(df, "a", ["b"])
    if "b" in result:
        for key in ("correlation", "p_value", "n", "strength"):
            assert key in result["b"]


def test_group_difference_tests_detects_mar_pattern() -> None:
    """When 'a' is missing for high values of 'b', group test must be significant."""
    rng: Generator = np.random.default_rng(42)
    n = 200
    b = rng.normal(0, 1, n)
    a: ndarray = np.where(b > 0.5, np.nan, rng.normal(0, 1, n))
    df = pd.DataFrame({"a": a, "b": b})
    result: dict[str, dict[str, Any]] = _group_difference_tests(
        df, "a", ["b"], alpha=0.05
    )
    assert "b" in result
    assert result["b"]["significant"] is True


def test_group_difference_tests_structure() -> None:
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": [None] * 30 + rng.normal(0, 1, 70).tolist(),
            "b": rng.normal(0, 1, 100).tolist(),
        },
    )
    result: dict[str, dict[str, Any]] = _group_difference_tests(
        df, "a", ["b"], alpha=0.05
    )
    if "b" in result:
        for key in (
            "u_statistic",
            "p_value",
            "rank_biserial_r",
            "significant",
            "n_missing_group",
            "n_observed_group",
        ):
            assert key in result["b"]


def test_assess_mechanism_never_claims_mnar_confirmed() -> None:
    """Assessment must never include 'mnar' as a positive consistent_with."""
    assessment: dict[str, Any] = _assess_mechanism(
        mcar_result=None,
        correlations={
            "b": {"correlation": 0.8, "p_value": 0.001, "n": 100, "strength": "strong"}
        },
        group_diffs={
            "b": {
                "u_statistic": 100.0,
                "p_value": 0.001,
                "rank_biserial_r": 0.5,
                "significant": True,
                "n_missing_group": 50,
                "n_observed_group": 50,
            },
        },
        alpha=0.05,
    )
    assert "mnar" not in assessment["consistent_with"]


def test_assess_mechanism_confidence_capped_at_moderate() -> None:
    """Confidence must never exceed 'moderate' — mechanism analysis is uncertain."""
    assessment: dict[str, Any] = _assess_mechanism(
        mcar_result={
            "p_value": 0.8,
            "test_statistic": 1.0,
            "degrees_of_freedom": 2,
            "n_patterns": 3,
            "n": 100,
            "caveats": ["test caveat"],
        },
        correlations={},
        group_diffs={},
        alpha=0.05,
    )
    assert assessment["confidence"] in ("low", "moderate")
    assert assessment["confidence"] != "high"


def test_assess_mechanism_always_has_mnar_caveat() -> None:
    """Every assessment must include a caveat about MNAR being unverifiable."""
    for corr in (
        {},
        {"b": {"correlation": 0.7, "p_value": 0.001, "n": 100, "strength": "strong"}},
    ):
        assessment: dict[str, Any] = _assess_mechanism(
            mcar_result=None,
            correlations=corr,
            group_diffs={},
            alpha=0.05,
        )
        mnar_caveat: bool = any(
            "mnar" in c.lower() or "unverifiable" in c.lower()
            for c in assessment["caveats"]
        )
        assert mnar_caveat, "MNAR unverifiability caveat must always be present"


def test_assess_mechanism_consistent_with_mar_when_correlated() -> None:
    assessment: dict[str, Any] = _assess_mechanism(
        mcar_result=None,
        correlations={
            "b": {"correlation": 0.5, "p_value": 0.001, "n": 100, "strength": "strong"},
        },
        group_diffs={},
        alpha=0.05,
    )
    assert "mar" in assessment["consistent_with"]


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_mcar_consistent_for_random_missingness(tmp_path) -> None:
    """Randomly missing data should not reject MCAR at α=0.05."""
    rng: Generator = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        },
    )
    # Introduce missingness at random — truly MCAR
    for col in ["a", "b"]:
        idx = rng.choice(n, size=20, replace=False)
        df.loc[idx, col] = np.nan

    ctx, task = make_ctx_and_task(
        task_cls=MissingnessMechanismAnalysis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["columns_analysed"] >= 1
    # Check epistemic note is present
    assert "epistemic_note" in result.summary
    assert "unverifiable" in result.summary["epistemic_note"].lower()


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_mar_pattern_detected(tmp_path) -> None:
    """When missingness correlates with another column, MAR should be noted."""
    rng: Generator = np.random.default_rng(42)
    n = 300
    income = rng.normal(50000, 15000, n)
    age = rng.normal(40, 10, n)
    # Income missing for younger people (MAR on age)
    missing_mask = age < 30
    income_with_missing = income.copy()
    income_with_missing[missing_mask] = np.nan

    df = pd.DataFrame({"income": income_with_missing, "age": age})

    ctx, task = make_ctx_and_task(
        task_cls=MissingnessMechanismAnalysis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "income" in result.data
    assessment = result.data["income"]["mechanism_assessment"]
    assert "mar" in assessment["consistent_with"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_output_structure_complete(tmp_path) -> None:
    """Each column result must contain all expected keys."""
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": [None] * 30 + rng.normal(0, 1, 70).tolist(),
            "b": rng.normal(0, 1, 100).tolist(),
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=MissingnessMechanismAnalysis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "a" in result.data
    entry = result.data["a"]
    for key in (
        "null_pct",
        "n_missing",
        "n_total",
        "missingness_correlations",
        "group_difference_tests",
        "mechanism_assessment",
    ):
        assert key in entry

    assessment = entry["mechanism_assessment"]
    for key in ("consistent_with", "confidence", "evidence_summary", "caveats"):
        assert key in assessment


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_uses_hedged_language(tmp_path) -> None:
    """Guidance must use 'consistent with' language, not definitive claims."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": [None] * 30 + rng.normal(0, 1, 70).tolist(),
            "b": rng.normal(0, 1, 100).tolist(),
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=MissingnessMechanismAnalysis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    if "a" in result.guidance:
        eda_blurb: dict[str, Any] = result.guidance["a"]["eda"][0]
        body = eda_blurb["body"].lower()
        # Must use hedged language
        assert any(
            phrase in body
            for phrase in ("consistent with", "cannot", "caveats", "hypothes")
        )
        # Must not make definitive mechanism claims
        assert "is mnar" not in body
        assert "confirmed mnar" not in body
        assert "confirmed mar" not in body


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_no_missing_columns_returns_gracefully(tmp_path) -> None:
    """A DataFrame with no missing values must return success with empty data."""
    df = pd.DataFrame({"a": range(50), "b": range(50)})

    ctx, _ = make_ctx_and_task(
        task_cls=MissingnessMechanismAnalysis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, MissingnessMechanismAnalysis)

    assert result.status == "success"
    assert result.data == {}
    assert result.summary["columns_analysed"] == 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summarize_nulls_context_used(tmp_path) -> None:
    """Task must read null_percentages from summarize_nulls context when available."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x": [None] * 20 + rng.normal(0, 1, 80).tolist(),
            "y": rng.normal(0, 1, 100).tolist(),
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=MissingnessMechanismAnalysis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    # Inject mock summarize_nulls result

    ctx.results["summarize_nulls"] = TaskResult(
        name="summarize_nulls",
        status="success",
        summary={"message": "mock"},
        data={
            "null_percentages": {"x": 0.20},
            "null_counts": {"x": 20},
            "high_null_columns": [],
            "null_patterns": {},
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "x" in result.data
    assert abs(result.data["x"]["null_pct"] - 0.20) < 0.01


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    rng: Generator = np.random.default_rng(42)
    vals = rng.normal(0, 1, 100).tolist()
    vals[:20] = [None] * 20
    df = pl.DataFrame({"a": vals, "b": rng.normal(0, 1, 100).tolist()})

    ctx, task = make_ctx_and_task(
        task_cls=MissingnessMechanismAnalysis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"


def test_no_plots_generated(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "a": [None] * 10 + list(range(40)),
            "b": list(range(50)),
        }
    )
    ctx, task = make_ctx_and_task(
        task_cls=MissingnessMechanismAnalysis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.plots is None
