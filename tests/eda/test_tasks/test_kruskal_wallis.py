# tests/eda/test_tasks/test_kruskal_wallis.py

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.kruskal_wallis import KruskalWallis
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from collections.abc import Generator

    from dsbf.eda.task_result import TaskResult


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_significant_difference_detected(tmp_path) -> None:
    """Groups with clearly different distributions must produce a significant H."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "value": np.concatenate(
                [
                    rng.exponential(1, 100),
                    rng.exponential(10, 100),  # very different scale
                ],
            ),
            "group": ["A"] * 100 + ["B"] * 100,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=KruskalWallis,
        current_df=df,
        task_overrides={"alpha": 0.05},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "value|group" in result.data
    assert result.data["value|group"]["p_value"] < 0.05
    assert result.data["value|group"]["significant"] is True


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_result_structure_complete(tmp_path) -> None:
    """Each result entry must contain all expected keys."""
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)]),
            "cat": ["A"] * 50 + ["B"] * 50,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=KruskalWallis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "cat": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    entry = result.data["x|cat"]
    for key in (
        "h_statistic",
        "p_value",
        "n_groups",
        "n_total",
        "significant",
        "alpha",
    ):
        assert key in entry


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_non_normal_groups_handled(tmp_path) -> None:
    """Kruskal-Wallis must work correctly on skewed, non-normal distributions."""
    rng: Generator = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "amount": np.concatenate(
                [
                    rng.exponential(1, 150),
                    rng.exponential(5, 150),
                ],
            ),
            "tier": ["low"] * 150 + ["high"] * 150,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=KruskalWallis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"amount": "continuous", "tier": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["amount|tier"]["p_value"] < 0.05


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_significant_result(tmp_path) -> None:
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "score": np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100)]),
            "group": ["A"] * 100 + ["B"] * 100,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=KruskalWallis,
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
        task_cls=KruskalWallis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "cat": "categorical"})
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.summary["pair_count"] == len(result.data)
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
        task_cls=KruskalWallis,
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
        task_cls=KruskalWallis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "g": "categorical"})
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.plots is None
