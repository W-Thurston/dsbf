# tests/eda/test_tasks/test_compute_entropy.py

from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.compute_entropy import ComputeEntropy
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult, WarningDetail


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_entropy_computed_for_categorical_column(tmp_path) -> None:
    """Entropy must be positive for categorical column with multiple distinct values."""
    df = pd.DataFrame(
        {
            "cat": ["a", "a", "b", "b", "b", "c", "c", "c", "c"],
            "num": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # numeric — must be excluded
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result.status == "success"
    assert result.data is not None
    assert "cat" in result.data
    assert result.data["cat"] > 0.0
    assert "num" not in result.data


def test_entropy_is_zero_for_constant_column(tmp_path) -> None:
    """A column with a single unique value must have entropy 0.0."""
    df = pd.DataFrame({"constant": ["x"] * 100})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result.status == "success"
    assert result.data is not None
    assert result.data.get("constant") == 0.0


def test_entropy_skips_all_null_column(tmp_path) -> None:
    """A column with all missing values must be excluded from output without error."""
    df = pd.DataFrame({"empty": [None] * 10})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result.status == "success"
    assert result.data is not None
    assert "empty" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_entropy_higher_for_uniform_distribution(tmp_path) -> None:
    """A uniform distribution must have higher entropy than a skewed one."""
    df = pd.DataFrame(
        {
            "uniform": ["a", "b", "c", "d"] * 25,  # 4 equal groups
            "skewed": ["a"] * 90 + ["b"] * 10,  # 90/10 split
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result.status == "success"
    assert "uniform" in result.data
    assert "skewed" in result.data
    assert result.data["uniform"] > result.data["skewed"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_entropy_metadata_fields_populated(tmp_path) -> None:
    """excluded_columns and column_types must be present in metadata."""
    df = pd.DataFrame(
        {
            "col1": ["a", "b", "c", "a"],
            "col2": [1, 2, 3, 4],  # numeric — must be excluded
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result.status == "success"
    metadata: dict[str, Any] = result.metadata
    assert "excluded_columns" in metadata
    assert "col2" in metadata["excluded_columns"]
    assert "column_types" in metadata
    assert "col1" in metadata["column_types"]
    assert "col2" in metadata["column_types"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_no_plots_generated(tmp_path) -> None:
    """Entropy task must not generate plots — rendering is owned by
    generate_univariate_plots and generate_dataset_summary_plots."""
    df = pd.DataFrame({"cat": ["a", "b", "c"] * 10})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result.status == "success"
    assert result.plots is None


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Entropy must be computed correctly for a Polars DataFrame."""
    df = pl.DataFrame(
        {
            "color": ["red", "red", "blue", "green", "blue", "green", "green"],
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result.status == "success"
    assert result.data is not None
    assert "color" in result.data
    assert result.data["color"] > 0.0


def test_reliability_warning_on_low_n(tmp_path) -> None:
    """A heuristic_caution warning must be emitted when N < 30."""
    df = pd.DataFrame({"cat": ["a", "b", "c", "a", "b"]})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result.status == "success"
    assert result.reliability_warnings is not None
    caution: dict[str, WarningDetail] = result.reliability_warnings.get(
        "heuristic_caution", {}
    )
    assert "low_row_count_entropy" in caution
