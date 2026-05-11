# tests/eda/test_tasks/test_one_way_anova.py
#
# Design principles:
#   - All DataFrames use linspace-repeated values so infer_types classifies
#     numeric columns as "continuous" (unique_ratio ~0.125, not id-like).
#   - Semantic types are injected via ctx.set_metadata before running
#     so infer_types is skipped and the injected classification is preserved.
#   - run_task_with_dependencies is used for all integration tests.
#   - Unit tests for _apply_correction cover the pure function independently.

import random
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.one_way_anova import OneWayANOVA, _apply_correction
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def _two_group_df(
    n_per_group: int = 50,
    low: float = 0.0,
    high: float = 5.0,
    n_unique: int = 25,
    group_col: str = "group",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Two-group DataFrame where the numeric column has enough unique values
    (unique_ratio = n_unique / total < 0.9) for infer_types → continuous.
    """
    lo = np.linspace(low, low + 1, n_unique).tolist() * (n_per_group // n_unique + 1)
    hi = np.linspace(high, high + 1, n_unique).tolist() * (n_per_group // n_unique + 1)
    return pd.DataFrame(
        {
            value_col: lo[:n_per_group] + hi[:n_per_group],
            group_col: ["A"] * n_per_group + ["B"] * n_per_group,
        }
    )


# ── Pure-function unit tests ───────────────────────────────────────────────────


def test_correction_none_returns_unchanged() -> None:
    p = [0.01, 0.04, 0.20]
    assert _apply_correction(p, "none") == p


def test_correction_bonferroni_scales_by_n() -> None:
    p = [0.01, 0.02, 0.10]
    c = _apply_correction(p, "bonferroni")
    assert abs(c[0] - 0.03) < 1e-6
    assert abs(c[1] - 0.06) < 1e-6
    assert abs(c[2] - 0.30) < 1e-6


def test_correction_bonferroni_caps_at_one() -> None:
    assert all(v <= 1.0 for v in _apply_correction([0.5, 0.6, 0.7], "bonferroni"))


def test_correction_fdr_bh_small_p_survives() -> None:
    c = _apply_correction([0.001] + [0.9] * 9, "fdr_bh")
    assert c[0] < 0.05
    assert all(v <= 1.0 for v in c)


def test_correction_single_value() -> None:
    for m in ("bonferroni", "fdr_bh", "none"):
        r = _apply_correction([0.03], m)
        assert len(r) == 1 and r[0] <= 1.0


def test_correction_fdr_bh_all_in_range() -> None:
    rng = random.Random(42)
    p = [rng.uniform(0, 1) for _ in range(20)]
    c = _apply_correction(p, "fdr_bh")
    assert all(0.0 <= v <= 1.0 for v in c) and len(c) == 20


# ── Integration tests ──────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_significant_group_difference_detected(tmp_path) -> None:
    """Large group separation must produce a significant ANOVA result."""
    df = _two_group_df(n_per_group=100, low=0.0, high=9.0)
    ctx, _ = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, OneWayANOVA)
    assert result.status == "success"
    assert "value|group" in result.data
    assert result.data["value|group"]["significant"] is True


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_corrected_fields_present(tmp_path) -> None:
    """Each result entry must have p_value, p_value_corrected, correction."""
    df = _two_group_df(n_per_group=100, low=0.0, high=9.0)
    ctx, _ = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, OneWayANOVA)
    assert result.status == "success"
    e = result.data["value|group"]
    assert "p_value" in e and "p_value_corrected" in e and "correction" in e
    assert e["correction"] == "fdr_bh"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_correction_none_p_values_equal(tmp_path) -> None:
    """With correction='none', raw and corrected p-values must be equal."""
    df = _two_group_df(n_per_group=50, low=0.0, high=5.0)
    ctx, _ = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"correction": "none"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, OneWayANOVA)
    assert result.status == "success"
    e = result.data["value|group"]
    assert e["p_value"] == e["p_value_corrected"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_n_tests_in_metadata(tmp_path) -> None:
    """n_tests in metadata must equal the number of result entries."""
    df = _two_group_df(n_per_group=50, value_col="x", group_col="cat")
    ctx, _ = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "cat": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, OneWayANOVA)
    assert result.status == "success"
    assert result.metadata["n_tests"] == len(result.data)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_correction_reported_in_summary(tmp_path) -> None:
    """The correction method must appear in result summary."""
    df = _two_group_df(n_per_group=50, value_col="x", group_col="cat")
    ctx, _ = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"correction": "bonferroni"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "cat": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, OneWayANOVA)
    assert result.status == "success"
    assert result.summary["correction"] == "bonferroni"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_three_groups_significant(tmp_path) -> None:
    """Three clearly separated groups must yield a significant result."""
    n = 33
    u = 11
    df = pd.DataFrame(
        {
            "value": (
                np.linspace(0, 1, u).tolist() * 3
                + np.linspace(9, 10, u).tolist() * 3
                + np.linspace(18, 19, u).tolist() * 3
            ),
            "group": ["A"] * n + ["B"] * n + ["C"] * n,
        }
    )
    ctx, _ = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, OneWayANOVA)
    assert result.status == "success"
    assert result.data["value|group"]["significant"] is True


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_significant_flag_uses_corrected_p(tmp_path) -> None:
    """significant flag must reflect corrected p-value, not raw."""
    df = _two_group_df(n_per_group=100, low=0.0, high=9.0)
    ctx, _ = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"alpha": 0.05, "correction": "fdr_bh"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, OneWayANOVA)
    assert result.status == "success"
    for e in result.data.values():
        assert e["significant"] == (e["p_value_corrected"] < 0.05)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames."""
    df = pl.from_pandas(_two_group_df(n_per_group=50))
    ctx, _ = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, OneWayANOVA)
    assert result.status == "success"
    assert "value|group" in result.data


def test_no_plots_generated(tmp_path) -> None:
    """Task must not generate static plot files."""
    df = _two_group_df(n_per_group=30, value_col="x", group_col="g")
    ctx, _ = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "g": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, OneWayANOVA)
    assert result.status == "success"
    assert result.plots is None
