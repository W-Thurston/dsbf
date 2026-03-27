# tests/eda/test_tasks/test_one_way_anova.py

import random
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.one_way_anova import OneWayANOVA, _apply_correction
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from collections.abc import Generator

    from dsbf.eda.task_result import TaskResult

# ── Unit tests for _apply_correction ─────────────────────────────────────────


def test_correction_none_returns_unchanged() -> None:
    p: list[float] = [0.01, 0.04, 0.20]
    assert _apply_correction(p, "none") == p


def test_correction_bonferroni_scales_by_n() -> None:
    p: list[float] = [0.01, 0.02, 0.10]
    corrected: list[float] = _apply_correction(p, "bonferroni")
    assert abs(corrected[0] - 0.03) < 1e-6
    assert abs(corrected[1] - 0.06) < 1e-6
    assert abs(corrected[2] - 0.30) < 1e-6


def test_correction_bonferroni_caps_at_one() -> None:
    p: list[float] = [0.5, 0.6, 0.7]
    corrected: list[float] = _apply_correction(p, "bonferroni")
    assert all(v <= 1.0 for v in corrected)


def test_correction_fdr_bh_small_p_survives() -> None:
    """A single tiny p-value among many large ones must survive BH correction."""
    p: list[float] = [0.001] + [0.9] * 9
    corrected: list[float] = _apply_correction(p, "fdr_bh")
    assert corrected[0] < 0.05
    assert all(v <= 1.0 for v in corrected)


def test_correction_single_value_unchanged() -> None:
    for method in ("bonferroni", "fdr_bh", "none"):
        result: list[float] = _apply_correction([0.03], method)
        assert len(result) == 1
        assert result[0] <= 1.0


def test_correction_fdr_bh_all_values_in_range() -> None:
    rng: Generator = random.Random(42)
    p: list[float] = [rng.uniform(0, 1) for _ in range(20)]
    corrected: list[float] = _apply_correction(p, "fdr_bh")
    assert all(0.0 <= v <= 1.0 for v in corrected)
    assert len(corrected) == len(p)


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_corrected_fields_present(tmp_path) -> None:
    """Each result entry must contain p_value, p_value_corrected, and correction."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100)]),
            "group": ["A"] * 100 + ["B"] * 100,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    entry = result.data["value|group"]
    assert "p_value" in entry
    assert "p_value_corrected" in entry
    assert "correction" in entry
    assert entry["correction"] == "fdr_bh"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_significant_flag_uses_corrected_p(tmp_path) -> None:
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100)]),
            "group": ["A"] * 100 + ["B"] * 100,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"alpha": 0.05, "correction": "fdr_bh"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    for entry in result.data.values():
        assert entry["significant"] == (entry["p_value_corrected"] < 0.05)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_correction_none_p_values_equal(tmp_path) -> None:
    """With correction='none', p_value_corrected must equal p_value."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)]),
            "group": ["A"] * 50 + ["B"] * 50,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"correction": "none"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    entry = result.data["value|group"]
    assert entry["p_value"] == entry["p_value_corrected"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_bonferroni_more_conservative_than_fdr(tmp_path) -> None:
    """Bonferroni corrected p-values must be >= FDR corrected p-values."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 1, 50)]),
            "b": np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 1, 50)]),
            "group": ["X"] * 50 + ["Y"] * 50,
        },
    )

    ctx_b, task_b = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"correction": "bonferroni"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx_b.set_metadata(
        "semantic_types", {"a": "continuous", "b": "continuous", "group": "categorical"}
    )
    result_b: TaskResult = ctx_b.run_task(task_b)

    ctx_f, task_f = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"correction": "fdr_bh"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx_f.set_metadata(
        "semantic_types", {"a": "continuous", "b": "continuous", "group": "categorical"}
    )
    result_f: TaskResult = ctx_f.run_task(task_f)

    assert result_b.status == "success"
    assert result_f.status == "success"
    for key in result_b.data:
        assert (
            result_b.data[key]["p_value_corrected"]
            >= result_f.data[key]["p_value_corrected"] - 1e-9
        )


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_n_tests_in_metadata(tmp_path) -> None:
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)]),
            "cat": ["A"] * 50 + ["B"] * 50,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "cat": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.metadata["n_tests"] == len(result.data)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_correction_reported_in_summary(tmp_path) -> None:
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)]),
            "cat": ["A"] * 50 + ["B"] * 50,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"correction": "bonferroni"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "cat": "categorical"})
    result: TaskResult = ctx.run_task(task)
    assert result.summary["correction"] == "bonferroni"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_significant_group_difference_detected(tmp_path) -> None:
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "value": np.concatenate(
                [
                    rng.normal(0, 1, 100),
                    rng.normal(10, 1, 100),
                    rng.normal(20, 1, 100),
                ],
            ),
            "group": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.data["value|group"]["significant"] is True


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    rng: Generator = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "value": np.concatenate(
                [rng.normal(0, 1, 50), rng.normal(5, 1, 50)]
            ).tolist(),
            "group": ["A"] * 50 + ["B"] * 50,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"


def test_no_plots_generated(tmp_path) -> None:
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, 30), rng.normal(5, 1, 30)]),
            "g": ["A"] * 30 + ["B"] * 30,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "g": "categorical"})
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.plots is None
