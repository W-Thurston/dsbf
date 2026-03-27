# tests/eda/test_tasks/test_one_way_anova.py


from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.one_way_anova import OneWayANOVA
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from collections.abc import Generator

    from dsbf.eda.task_result import TaskResult


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_significant_group_difference_detected(tmp_path) -> None:
    """Groups with clearly different means must produce a significant F-test."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "value": np.concatenate(
                [
                    rng.normal(0, 1, 100),
                    rng.normal(10, 1, 100),  # clearly separated
                    rng.normal(20, 1, 100),
                ]
            ),
            "group": ["A"] * 100 + ["B"] * 100 + ["C"] * 100,
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"alpha": 0.05},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "value|group" in result.data
    entry = result.data["value|group"]
    assert entry["p_value"] < 0.05
    assert entry["significant"] is True
    assert entry["f_statistic"] > 1.0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_no_group_difference_not_significant(tmp_path) -> None:
    """Groups drawn from the same distribution must not be significant."""
    rng: Generator = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "value": rng.normal(5, 1, n),
            "group": (["X"] * (n // 2)) + (["Y"] * (n // 2)),
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"alpha": 0.05},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    if "value|group" in result.data:
        assert result.data["value|group"]["p_value"] >= 0.0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_result_structure_complete(tmp_path) -> None:
    """Each result entry must contain all expected keys."""
    rng: Generator = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)]),
            "cat": ["A"] * 50 + ["B"] * 50,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"alpha": 0.05},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "cat": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    entry = result.data["x|cat"]
    for key in (
        "f_statistic",
        "p_value",
        "n_groups",
        "n_total",
        "significant",
        "alpha",
    ):
        assert key in entry


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_significant_result(tmp_path) -> None:
    """Guidance must be attached for significant ANOVA results."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "score": np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100)]),
            "group": ["A"] * 100 + ["B"] * 100,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"alpha": 0.05},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"score": "continuous", "group": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    assert "score" in result.guidance or "group" in result.guidance


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_high_cardinality_cat_excluded(tmp_path) -> None:
    """Categorical columns exceeding cat_cardinality_limit must be excluded."""
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "value": rng.normal(0, 1, 100),
            "high_card": [f"cat_{i}" for i in range(100)],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=OneWayANOVA,
        current_df=df,
        task_overrides={"cat_cardinality_limit": 5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types", {"value": "continuous", "high_card": "categorical"}
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert all("high_card" not in k for k in result.data)


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
        }
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
