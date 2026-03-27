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
def test_separated_groups_significant(tmp_path) -> None:
    """Clearly separated groups must produce a significant U test."""
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
        task_overrides={"alpha": 0.05, "min_group_n": 5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    key = "value|group|A_vs_B"
    assert key in result.data
    assert result.data[key]["p_value"] < 0.05
    assert result.data[key]["significant"] is True


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_rank_biserial_r_range(tmp_path) -> None:
    """Rank-biserial r must be in [-1, 1]."""
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
def test_result_structure_complete(tmp_path) -> None:
    """Each result entry must contain all expected keys."""
    rng: Generator = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)]),
            "flag": ["yes"] * 50 + ["no"] * 50,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=MannWhitneyU,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "flag": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    entry = next(iter(result.data.values()))
    for key in (
        "u_statistic",
        "p_value",
        "rank_biserial_r",
        "n_group_a",
        "n_group_b",
        "level_a",
        "level_b",
        "num_col",
        "cat_col",
        "significant",
        "alpha",
    ):
        assert key in entry


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_multi_level_produces_multiple_tests(tmp_path) -> None:
    """A 3-level categorical must produce 3 pairwise tests."""
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
        task_overrides={"cat_cardinality_limit": 10},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    # 3 levels → 3 pairs: A_vs_B, A_vs_C, B_vs_C
    assert len(result.data) == 3


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_significant_pair(tmp_path) -> None:
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "score": np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100)]),
            "group": ["A"] * 100 + ["B"] * 100,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=MannWhitneyU,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"score": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.guidance is not None


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_correct(tmp_path) -> None:
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
    assert result.summary["test_count"] == len(result.data)
    sig: int = sum(1 for v in result.data.values() if v["significant"])
    assert result.summary["significant_count"] == sig


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
