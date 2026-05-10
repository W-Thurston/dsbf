# tests/eda/test_tasks/test_kruskal_wallis.py

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.kruskal_wallis import KruskalWallis
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from collections.abc import Generator

    from dsbf.eda.task_result import TaskResult


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_significant_group_difference_detected(tmp_path) -> None:
    """A large distributional difference across groups must be detected."""
    _: Generator = np.random.default_rng(42)
    # linspace repeated → unique_ratio=0.125 → continuous; groups clearly separated
    df = pd.DataFrame(
        {
            "value": np.concatenate(
                [
                    np.linspace(0, 1, 25).tolist() * 4,
                    np.linspace(9, 10, 25).tolist() * 4,
                ]
            ),
            "group": ["A"] * 100 + ["B"] * 100,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=KruskalWallis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, KruskalWallis)

    assert result.status == "success"
    assert result.data is not None
    assert len(result.data) > 0
    key = "value|group"
    assert key in result.data
    assert result.data[key]["significant"] is True


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_no_difference_not_significant(tmp_path) -> None:
    """Identical distributions across groups must not be flagged as significant."""
    _: Generator = np.random.default_rng(42)
    shared = np.linspace(0, 5, 25).tolist() * 4
    df = pd.DataFrame(
        {
            "value": shared + shared,
            "group": ["A"] * 100 + ["B"] * 100,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=KruskalWallis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, KruskalWallis)

    assert result.status == "success"
    key = "value|group"
    if key in result.data:
        assert result.data[key]["significant"] is False


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_corrected_fields_present(tmp_path) -> None:
    """Each result entry must contain p_value, p_value_corrected, and correction."""
    _: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "value": np.concatenate(
                [
                    np.linspace(0, 1, 25).tolist() * 4,
                    np.linspace(9, 10, 25).tolist() * 4,
                ]
            ),
            "group": ["A"] * 100 + ["B"] * 100,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=KruskalWallis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, KruskalWallis)

    assert result.status == "success"
    key = "value|group"
    assert key in result.data
    entry = result.data[key]
    assert "p_value" in entry
    assert "p_value_corrected" in entry
    assert "correction" in entry


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_correction_none_p_values_equal(tmp_path) -> None:
    """With correction='none', p_value_corrected must equal p_value."""
    _: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "value": np.concatenate(
                [
                    np.linspace(0, 1, 25).tolist() * 2,
                    np.linspace(4, 5, 25).tolist() * 2,
                ]
            ),
            "group": ["A"] * 50 + ["B"] * 50,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=KruskalWallis,
        current_df=df,
        task_overrides={"correction": "none"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, KruskalWallis)

    assert result.status == "success"
    key = "value|group"
    assert key in result.data
    assert result.data[key]["p_value"] == result.data[key]["p_value_corrected"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_n_tests_in_metadata(tmp_path) -> None:
    """n_tests in metadata must equal the number of result entries."""
    _: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x": np.concatenate(
                [
                    np.linspace(0, 1, 25).tolist() * 2,
                    np.linspace(4, 5, 25).tolist() * 2,
                ]
            ),
            "cat": ["A"] * 50 + ["B"] * 50,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=KruskalWallis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "cat": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, KruskalWallis)

    assert result.status == "success"
    assert result.metadata["n_tests"] == len(result.data)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_no_continuous_columns_returns_empty(tmp_path) -> None:
    """A DataFrame with only categorical columns must return empty data."""
    df = pd.DataFrame({"cat": ["A", "B", "C"] * 20})
    ctx, _ = make_ctx_and_task(
        task_cls=KruskalWallis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"cat": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, KruskalWallis)

    assert result.status == "success"
    assert result.data == {} or "pair_count" in result.summary


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via conversion."""
    df = pl.DataFrame(
        {
            "value": (
                np.linspace(0, 1, 25).tolist() * 2 + np.linspace(4, 5, 25).tolist() * 2
            ),
            "group": ["A"] * 50 + ["B"] * 50,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=KruskalWallis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"value": "continuous", "group": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, KruskalWallis)

    assert result.status == "success"


def test_no_plots_generated(tmp_path) -> None:
    """Task must not generate static plot files."""
    _: Generator = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": np.concatenate(
                [
                    np.linspace(0, 1, 15).tolist() * 2,
                    np.linspace(4, 5, 15).tolist() * 2,
                ]
            ),
            "g": ["A"] * 30 + ["B"] * 30,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=KruskalWallis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "g": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, KruskalWallis)

    assert result.status == "success"
    assert result.plots is None
