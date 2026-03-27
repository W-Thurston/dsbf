# tests/eda/test_tasks/test_mann_whitney_u.py

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.mann_whitney_u import MannWhitneyU
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from collections.abc import Generator

    from dsbf.eda.task_result import TaskResult


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
        task_cls=MannWhitneyU,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    key = "value|group|A_vs_B"
    assert key in result.data
    entry = result.data[key]
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
        task_cls=MannWhitneyU,
        current_df=df,
        task_overrides={"alpha": 0.05},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    for entry in result.data.values():
        assert entry["significant"] == (entry["p_value_corrected"] < 0.05)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_correction_none_p_values_equal(tmp_path) -> None:
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)]),
            "group": ["A"] * 50 + ["B"] * 50,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=MannWhitneyU,
        current_df=df,
        task_overrides={"correction": "none"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    for entry in result.data.values():
        assert entry["p_value"] == entry["p_value_corrected"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_separated_groups_significant(tmp_path) -> None:
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100)]),
            "group": ["A"] * 100 + ["B"] * 100,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=MannWhitneyU,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["value|group|A_vs_B"]["significant"] is True


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_rank_biserial_r_range(tmp_path) -> None:
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, 60), rng.normal(3, 1, 60)]),
            "cat": ["low"] * 60 + ["high"] * 60,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=MannWhitneyU,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "cat": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    for entry in result.data.values():
        assert -1.0 <= entry["rank_biserial_r"] <= 1.0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_multi_level_correction_applied_across_all_pairs(tmp_path) -> None:
    """Correction must be applied across all 3 level-pairs, not per-pair."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x": np.concatenate(
                [
                    rng.normal(0, 1, 50),
                    rng.normal(5, 1, 50),
                    rng.normal(10, 1, 50),
                ],
            ),
            "group": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=MannWhitneyU,
        current_df=df,
        task_overrides={"cat_cardinality_limit": 10, "correction": "fdr_bh"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert len(result.data) == 3  # A_vs_B, A_vs_C, B_vs_C
    assert result.metadata["n_tests"] == 3
    # All entries must share the same correction
    for entry in result.data.values():
        assert entry["correction"] == "fdr_bh"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_n_tests_in_metadata(tmp_path) -> None:
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)]),
            "cat": ["A"] * 50 + ["B"] * 50,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=MannWhitneyU,
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
        task_cls=MannWhitneyU,
        current_df=df,
        task_overrides={"correction": "bonferroni"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "cat": "categorical"})
    result: TaskResult = ctx.run_task(task)
    assert result.summary["correction"] == "bonferroni"


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
        task_cls=MannWhitneyU,
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
        task_cls=MannWhitneyU,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "g": "categorical"})
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.plots is None
