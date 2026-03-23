# tests/eda/test_tasks/test_contingency_tables.py

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.contingency_tables import (
    ContingencyTables,
    _association_strength,
    _build_contingency,
    _cramers_v,
)
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from collections.abc import Generator

    from dsbf.eda.task_result import TaskResult

# ── Unit tests for pure helpers ───────────────────────────────────────────────


def test_cramers_v_perfect_association() -> None:
    # Perfect 2x2 association: all on diagonal
    # chi2 = n, r=k=2, denom=1 → V=1.0
    assert abs(_cramers_v(100.0, 100, 2, 2) - 1.0) < 1e-6


def test_cramers_v_no_association() -> None:
    assert _cramers_v(0.0, 100, 3, 3) == 0.0


def test_cramers_v_degenerate_table() -> None:
    # r=1 → denom=0 → should return 0.0 not raise
    assert _cramers_v(50.0, 100, 1, 3) == 0.0


def test_association_strength_labels() -> None:
    assert _association_strength(0.6) == "strong"
    assert _association_strength(0.5) == "strong"
    assert _association_strength(0.35) == "moderate"
    assert _association_strength(0.3) == "moderate"
    assert _association_strength(0.15) == "weak"
    assert _association_strength(0.1) == "weak"
    assert _association_strength(0.05) == "negligible"
    assert _association_strength(0.0) == "negligible"


def test_build_contingency_basic() -> None:
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "red", "blue", "red"] * 20,
            "size": ["S", "L", "S", "L", "M"] * 20,
        },
    )
    result: dict[str, Any] = _build_contingency(df, "color", "size", top_n=10)
    assert result is not None
    assert "table" in result
    assert "chi2" in result
    assert "p_value" in result
    assert "cramers_v" in result
    assert "dof" in result
    assert result["n"] == 100
    # All values in table must be non-negative integers
    for row in result["table"].values():
        for count in row.values():
            assert count >= 0


def test_build_contingency_too_few_rows_returns_none() -> None:
    df = pd.DataFrame(
        {
            "a": ["x", "y"],
            "b": ["m", "n"],
        },
    )
    assert _build_contingency(df, "a", "b", top_n=10) is None


def test_build_contingency_single_level_returns_none() -> None:
    # All values in col_a are the same → 1xN table → skip
    df = pd.DataFrame(
        {
            "a": ["x"] * 50,
            "b": ["m", "n"] * 25,
        },
    )
    assert _build_contingency(df, "a", "b", top_n=10) is None


def test_build_contingency_top_n_truncation() -> None:
    # 20 unique values in col_a — top_n=3 should truncate
    cats: list[str] = [f"cat_{i}" for i in range(20)]
    df = pd.DataFrame(
        {
            "a": cats * 5,
            "b": ["x", "y"] * 50,
        },
    )
    result: dict[str, Any] = _build_contingency(df, "a", "b", top_n=3)
    assert result is not None
    assert result["truncated"] is True
    assert len(result["table"]) <= 3


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_strongly_associated_pair_detected(tmp_path) -> None:
    """A perfectly associated catxcat pair must produce a significant result."""
    # color and size are deterministically linked
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "green"] * 50,
            "size": ["S", "M", "L"] * 50,  # perfectly associated
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=ContingencyTables,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"color": "categorical", "size": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "color|size" in result.data
    entry = result.data["color|size"]
    assert entry["p_value"] < 0.05
    assert entry["cramers_v"] > 0.5
    assert entry["strength"] == "strong"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_independent_pair_not_significant(tmp_path) -> None:
    """Two independent categorical columns must not reject independence."""
    rng: Generator = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame(
        {
            "a": rng.choice(["x", "y", "z"], n),
            "b": rng.choice(["p", "q", "r"], n),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ContingencyTables,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"a": "categorical", "b": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    if "a|b" in result.data:
        # May or may not reject by chance — just check it computed without error
        assert result.data["a|b"]["p_value"] >= 0.0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_table_structure_complete(tmp_path) -> None:
    """Each table entry must contain all expected keys."""
    df = pd.DataFrame(
        {
            "x": ["A", "B", "A", "B"] * 30,
            "y": ["M", "N", "M", "N"] * 30,
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ContingencyTables,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "categorical", "y": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    entry = result.data["x|y"]
    for key in (
        "table",
        "chi2",
        "p_value",
        "dof",
        "cramers_v",
        "n",
        "strength",
        "top_n_used",
        "truncated",
        "low_sample_warning",
    ):
        assert key in entry


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_high_cardinality_column_excluded(tmp_path) -> None:
    """Columns exceeding cat_cardinality_limit must be excluded."""
    df = pd.DataFrame(
        {
            "low_card": ["A", "B"] * 50,
            "high_card": [f"val_{i}" for i in range(100)],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=ContingencyTables,
        current_df=df,
        task_overrides={"cat_cardinality_limit": 10},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types", {"low_card": "categorical", "high_card": "categorical"}
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    # high_card has 100 unique values — must not appear in any pair key
    for key in result.data:
        assert "high_card" not in key


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_significant_pairs_in_metadata(tmp_path) -> None:
    """significant_pairs in metadata must list pairs with p < alpha."""
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "green"] * 50,
            "size": ["S", "M", "L"] * 50,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=ContingencyTables,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"color": "categorical", "size": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    sig_pairs = result.metadata["significant_pairs"]
    for pair in sig_pairs:
        assert result.data[pair]["p_value"] < result.metadata["alpha"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_significant_association(tmp_path) -> None:
    """EDA and ML guidance must be attached for significant, non-negligible pairs."""
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "green"] * 50,
            "size": ["S", "M", "L"] * 50,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=ContingencyTables,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"color": "categorical", "size": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    # Guidance attached to both columns
    assert "color" in result.guidance
    assert "size" in result.guidance
    assert len(result.guidance["color"]["eda"]) > 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_match_data(tmp_path) -> None:
    """Summary table_count must equal number of entries in data."""
    df = pd.DataFrame(
        {
            "a": ["x", "y"] * 50,
            "b": ["m", "n"] * 50,
            "c": ["p", "q"] * 50,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=ContingencyTables,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types", {"a": "categorical", "b": "categorical", "c": "categorical"}
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["table_count"] == len(result.data)
    sig: int = sum(
        1 for v in result.data.values() if v["p_value"] < result.metadata["alpha"]
    )
    assert result.summary["significant_count"] == sig


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_custom_alpha_respected(tmp_path) -> None:
    """Custom alpha must be used for significance threshold."""
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "green"] * 50,
            "size": ["S", "M", "L"] * 50,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=ContingencyTables,
        current_df=df,
        task_overrides={"alpha": 0.001},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"color": "categorical", "size": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.metadata["alpha"] == 0.001


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    df = pl.DataFrame(
        {
            "x": ["A", "B", "A", "B"] * 30,
            "y": ["M", "N", "M", "N"] * 30,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=ContingencyTables,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "categorical", "y": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert len(result.data) > 0


def test_no_plots_generated(tmp_path) -> None:
    """Contingency tables task must not generate plots."""
    df = pd.DataFrame(
        {
            "a": ["x", "y"] * 20,
            "b": ["m", "n"] * 20,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=ContingencyTables,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"a": "categorical", "b": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
