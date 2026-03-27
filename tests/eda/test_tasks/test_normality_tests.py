# tests/eda/test_tasks/test_normality_tests.py

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import jarque_bera, shapiro

from dsbf.eda.tasks.normality_tests import (
    NormalityTests,
    _run_jarque_bera,
    _run_ks,
    _run_shapiro,
    _verdict,
)
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from collections.abc import Generator

    from dsbf.eda.task_result import TaskResult

# ── Unit tests for pure helper functions ──────────────────────────────────────


def test_verdict_normal() -> None:
    assert _verdict(0.10, 0.05) == "normal"
    assert _verdict(0.05, 0.05) == "normal"  # boundary - p == alpha is normal


def test_verdict_non_normal() -> None:
    assert _verdict(0.04, 0.05) == "non_normal"
    assert _verdict(0.001, 0.05) == "non_normal"


def test_verdict_inconclusive() -> None:
    assert _verdict(None, 0.05) == "inconclusive"


def test_run_shapiro_matches_scipy() -> None:
    rng: Generator = np.random.default_rng(42)
    values = rng.normal(0, 1, 100)
    result: dict = _run_shapiro(values)
    stat_expected, p_expected = shapiro(values)
    assert result["test"] == "shapiro_wilk"
    assert abs(result["statistic"] - float(stat_expected)) < 1e-6
    assert abs(result["p_value"] - float(p_expected)) < 1e-6


def test_run_ks_returns_expected_keys() -> None:
    rng: Generator = np.random.default_rng(42)
    values = rng.normal(0, 1, 200)
    result: dict = _run_ks(values)
    assert result["test"] == "ks_normal"
    assert "statistic" in result
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0


def test_run_ks_zero_variance_returns_none() -> None:
    result: dict = _run_ks(np.array([5.0] * 100))
    assert result["statistic"] is None
    assert result["p_value"] is None


def test_run_jarque_bera_matches_scipy() -> None:
    rng: Generator = np.random.default_rng(0)
    values = rng.exponential(2.0, 200)
    result: dict = _run_jarque_bera(values)
    stat_expected, p_expected = jarque_bera(values)
    assert result["test"] == "jarque_bera"
    assert abs(result["statistic"] - float(stat_expected)) < 1e-6
    assert abs(result["p_value"] - float(p_expected)) < 1e-6


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_normal_distribution_passes(tmp_path) -> None:
    """A large normally distributed sample must not reject normality."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"normal": rng.normal(0, 1, 500)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, NormalityTests)

    assert result.status == "success"
    assert "normal" in result.data
    # Shapiro-Wilk on 500 samples from N(0,1) should pass
    assert result.data["normal"]["overall_verdict"] == "normal"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_exponential_distribution_rejected(tmp_path) -> None:
    """A heavily skewed distribution must reject normality."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"skewed": rng.exponential(scale=1.0, size=500)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, NormalityTests)

    assert result.status == "success"
    assert result.data["skewed"]["overall_verdict"] == "non_normal"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_shapiro_used_for_small_sample(tmp_path) -> None:
    """Shapiro-Wilk must be used as primary test when n ≤ 5000."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.normal(0, 1, 100)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, NormalityTests)

    assert result.status == "success"
    assert result.data["x"]["primary_test"]["test"] == "shapiro_wilk"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_ks_used_for_large_sample(tmp_path) -> None:
    """KS test must be used as primary test when n > 5000."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.normal(0, 1, 6_000)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, NormalityTests)

    assert result.status == "success"
    assert result.data["x"]["primary_test"]["test"] == "ks_normal"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_jarque_bera_always_present(tmp_path) -> None:
    """Jarque-Bera result must always be present regardless of sample size."""
    rng: Generator = np.random.default_rng(42)

    # Small sample - uses Shapiro-Wilk as primary
    df_small = pd.DataFrame({"x": rng.normal(0, 1, 50)})
    ctx_s, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df_small,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result_small: TaskResult = run_task_with_dependencies(ctx_s, NormalityTests)

    # Large sample - uses KS as primary
    df_large = pd.DataFrame({"x": rng.normal(0, 1, 8_000)})
    ctx_l, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df_large,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result_large: TaskResult = run_task_with_dependencies(ctx_l, NormalityTests)

    for result in (result_small, result_large):
        assert result.status == "success"
        assert "jarque_bera" in result.data["x"]
        assert result.data["x"]["jarque_bera"]["test"] == "jarque_bera"
        assert result.data["x"]["jarque_bera"]["p_value"] is not None


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_column_below_min_n_skipped(tmp_path) -> None:
    """Columns with fewer than 8 non-null values must be skipped."""
    df = pd.DataFrame(
        {
            "tiny": [1.0, 2.0, 3.0] + [None] * 97,
            "normal": list(range(100)),
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, NormalityTests)

    assert result.status == "success"
    assert "tiny" not in result.data
    assert "normal" in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_custom_alpha_respected(tmp_path) -> None:
    """A strict alpha=0.01 must produce fewer non-normal verdicts than alpha=0.10."""
    rng: Generator = np.random.default_rng(42)
    # Mild non-normality - will reject at alpha=0.10 but may pass at alpha=0.01
    df = pd.DataFrame({"x": rng.lognormal(0, 0.3, 300)})

    ctx_strict, task_strict = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        task_overrides={"alpha": 0.001},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result_strict: TaskResult = ctx_strict.run_task(task_strict)

    ctx_loose, task_loose = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        task_overrides={"alpha": 0.99},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result_loose: TaskResult = ctx_loose.run_task(task_loose)

    assert result_strict.status == "success"
    assert result_loose.status == "success"
    # At alpha=0.99 almost everything rejects; at alpha=0.001 much less
    assert (
        result_loose.data["x"]["overall_verdict"] == "non_normal"
        or result_strict.data["x"]["overall_verdict"] == "normal"
    )


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_non_normal_column(tmp_path) -> None:
    """EDA and ML guidance must be attached for non-normal columns."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"skewed": rng.exponential(1.0, 500)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, NormalityTests)

    assert result.status == "success"
    assert result.guidance is not None
    assert "skewed" in result.guidance
    assert len(result.guidance["skewed"]["eda"]) > 0
    assert len(result.guidance["skewed"]["ml"]) > 0
    # ML guidance must include transform action chips
    ml_actions = result.guidance["skewed"]["ml"][0]["actions"]
    assert any(a["action"] == "transform" for a in ml_actions)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_no_guidance_for_normal_column(tmp_path) -> None:
    """No guidance must be emitted for columns that pass normality tests."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"normal": rng.normal(0, 1, 500)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, NormalityTests)

    assert result.status == "success"
    if result.data["normal"]["overall_verdict"] == "normal":
        assert result.guidance is None or "normal" not in (result.guidance or {})


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_correct(tmp_path) -> None:
    """non_normal_count in summary must equal columns with non_normal verdict."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "normal": rng.normal(0, 1, 500),
            "skewed": rng.exponential(1.0, 500),
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, NormalityTests)

    assert result.status == "success"
    non_normal_in_data: int = sum(
        1 for v in result.data.values() if v["overall_verdict"] == "non_normal"
    )
    assert result.summary["non_normal_count"] == non_normal_in_data
    assert result.summary["tested_count"] == len(result.data)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_output_structure_complete(tmp_path) -> None:
    """Every column result must contain all expected keys."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.normal(0, 1, 200)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, NormalityTests)

    assert result.status == "success"
    entry = result.data["x"]
    for key in (
        "n",
        "primary_test",
        "jarque_bera",
        "primary_verdict",
        "jb_verdict",
        "overall_verdict",
        "alpha",
    ):
        assert key in entry
    for key in ("test", "statistic", "p_value"):
        assert key in entry["primary_test"]
        assert key in entry["jarque_bera"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    rng: Generator = np.random.default_rng(42)
    df = pl.DataFrame({"x": rng.normal(0, 1, 100).tolist()})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, NormalityTests)

    assert result.status == "success"
    assert "x" in result.data


def test_no_plots_generated(tmp_path) -> None:
    """Normality tests task must not generate plots."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"a": rng.normal(0, 1, 100)})

    ctx, _ = make_ctx_and_task(
        task_cls=NormalityTests,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, NormalityTests)

    assert result.status == "success"
    assert result.plots is None
