# tests/eda/test_tasks/test_kendalls_tau.py


from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.kendalls_tau import KendallsTau, _strength
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from collections.abc import Generator

    from dsbf.eda.task_result import TaskResult

# ── Unit tests for _strength ──────────────────────────────────────────────────


def test_strength_strong() -> None:
    assert _strength(0.75) == "strong"
    assert _strength(-0.7) == "strong"


def test_strength_moderate() -> None:
    assert _strength(0.5) == "moderate"
    assert _strength(-0.4) == "moderate"


def test_strength_weak() -> None:
    assert _strength(0.3) == "weak"
    assert _strength(0.2) == "weak"


def test_strength_negligible() -> None:
    assert _strength(0.1) == "negligible"
    assert _strength(0.0) == "negligible"


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_perfectly_correlated_pair(tmp_path) -> None:
    """A perfectly monotonic pair must produce tau ≈ 1.0."""
    df = pd.DataFrame(
        {
            "x": list(range(1, 51)),
            "y": list(range(1, 51)),  # identical ordering
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=KendallsTau,
        current_df=df,
        task_overrides={"min_n": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "x|y" in result.data
    assert abs(result.data["x|y"]["tau"] - 1.0) < 1e-4


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_negatively_correlated_pair(tmp_path) -> None:
    """A perfectly reverse-monotonic pair must produce tau ≈ -1.0."""
    n = 50
    df = pd.DataFrame(
        {
            "x": list(range(n)),
            "y": list(range(n, 0, -1)),
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=KendallsTau,
        current_df=df,
        task_overrides={"min_n": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert abs(result.data["x|y"]["tau"] - (-1.0)) < 1e-4


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_independent_pair_near_zero(tmp_path) -> None:
    """Independent columns must produce tau close to zero."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 500),
            "b": rng.normal(0, 1, 500),
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=KendallsTau,
        current_df=df,
        task_overrides={"min_n": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"a": "continuous", "b": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert abs(result.data["a|b"]["tau"]) < 0.2


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_p_value_and_significant_flag(tmp_path) -> None:
    """A strongly correlated pair must have p_value < alpha and significant=True."""
    x: list[int] = list(range(1, 101))
    df = pd.DataFrame({"x": x, "y": x})

    ctx, task = make_ctx_and_task(
        task_cls=KendallsTau,
        current_df=df,
        task_overrides={"alpha": 0.05, "min_n": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    entry = result.data["x|y"]
    assert entry["p_value"] < 0.05
    assert entry["significant"] is True


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_result_structure_complete(tmp_path) -> None:
    """Each pair result must contain all expected keys."""
    df = pd.DataFrame({"a": range(30), "b": range(30)})

    ctx, task = make_ctx_and_task(
        task_cls=KendallsTau,
        current_df=df,
        task_overrides={"min_n": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"a": "continuous", "b": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    entry = result.data["a|b"]
    for key in ("tau", "p_value", "n", "strength", "significant"):
        assert key in entry


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_id_columns_excluded(tmp_path) -> None:
    """Columns typed as id must not appear in tau results."""
    df = pd.DataFrame(
        {
            "id": range(50),
            "feature": range(50),
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=KendallsTau,
        current_df=df,
        task_overrides={"min_n": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"id": "id", "feature": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    for key in result.data:
        assert "id" not in key.split("|")


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_significant_pair(tmp_path) -> None:
    """EDA guidance must be attached for significant pairs with |tau| >= 0.2."""
    x: list[int] = list(range(1, 51))
    df = pd.DataFrame({"x": x, "y": x})

    ctx, task = make_ctx_and_task(
        task_cls=KendallsTau,
        current_df=df,
        task_overrides={"alpha": 0.05, "min_n": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    # Both columns get guidance
    assert "x" in result.guidance
    assert "y" in result.guidance


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_correct(tmp_path) -> None:
    """Summary pair_count and significant_count must match data."""
    rng: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": range(50),
            "b": range(50),
            "c": rng.normal(0, 1, 50).tolist(),
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=KendallsTau,
        current_df=df,
        task_overrides={"alpha": 0.05, "min_n": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types", {"a": "continuous", "b": "continuous", "c": "continuous"}
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["pair_count"] == len(result.data)
    sig: int = sum(1 for v in result.data.values() if v["significant"])
    assert result.summary["significant_count"] == sig


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    df = pl.DataFrame({"x": list(range(30)), "y": list(range(30))})
    ctx, task = make_ctx_and_task(
        task_cls=KendallsTau,
        current_df=df,
        task_overrides={"min_n": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert len(result.data) > 0


def test_no_plots_generated(tmp_path) -> None:
    df = pd.DataFrame({"a": range(20), "b": range(20)})
    ctx, task = make_ctx_and_task(
        task_cls=KendallsTau,
        current_df=df,
        task_overrides={"min_n": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"a": "continuous", "b": "continuous"})
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.plots is None
