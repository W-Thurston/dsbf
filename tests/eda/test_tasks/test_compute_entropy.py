# tests/eda/test_tasks/test_compute_entropy.py

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.compute_entropy import ComputeEntropy
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_compute_entropy_expected_output(tmp_path):
    """
    Test that ComputeEntropy returns entropy values for text columns.
    """
    df = pd.DataFrame(
        {
            "cat": ["a", "a", "b", "b", "b", "c", "c", "c", "c"],
            "num": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Should be ignored
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result is not None, "No TaskResult returned"
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "cat" in result.data
    assert result.data["cat"] > 0.0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_compute_entropy_with_plots(tmp_path):
    """
    Test that barplots with entropy annotations are generated.
    """
    df = pd.DataFrame(
        {
            "color": [
                "red",
                "red",
                "blue",
                "blue",
                "blue",
                "green",
                "green",
                "green",
                "green",
            ],
            "constant": ["x"] * 9,
            "numeric": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Should be skipped
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result.status == "success"
    assert result.data is not None
    assert result.plots is not None

    for col in ["color", "constant"]:
        assert col in result.plots
        plot_entry = result.plots[col]

        # Validate static plot file
        static_path = plot_entry["static"]
        assert isinstance(static_path, Path)
        assert static_path.exists()
        static_path.unlink()

        # Validate interactive plot + annotations
        interactive = plot_entry["interactive"]
        assert isinstance(interactive, dict)
        assert "annotations" in interactive
        assert any("Entropy" in a for a in interactive["annotations"])


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_entropy_on_typical_categorical_column(tmp_path: Path) -> None:
    """
    Check that entropy is computed for a typical categorical column.
    """
    df = pd.DataFrame(
        {
            "cat": ["a", "a", "b", "b", "b", "c", "c", "c", "c"],
            "num": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Should be ignored
        }
    )

    ctx, task = make_ctx_and_task(
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


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_entropy_barplots_are_generated(tmp_path: Path) -> None:
    """
    Verify that barplots with entropy annotations are produced for eligible columns.
    """
    df = pd.DataFrame(
        {
            "color": [
                "red",
                "red",
                "blue",
                "blue",
                "blue",
                "green",
                "green",
                "green",
                "green",
            ],
            "constant": ["x"] * 9,
            "numeric": list(range(9)),  # Should be excluded
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeEntropy,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result.status == "success"
    assert result.plots is not None

    for col in ["color", "constant"]:
        assert col in result.plots, f"Missing plot entry for column: {col}"
        static_path = result.plots[col]["static"]
        assert isinstance(static_path, Path)
        assert static_path.exists()
        static_path.unlink()  # Clean up

        interactive = result.plots[col]["interactive"]
        assert isinstance(interactive, dict)
        assert any("Entropy" in a for a in interactive.get("annotations", []))


def test_entropy_on_constant_column() -> None:
    """
    Confirm that entropy is 0.0 for a column with a single unique value.
    """
    df = pd.DataFrame({"constant": ["x"] * 100})
    ctx, task = make_ctx_and_task(task_cls=ComputeEntropy, current_df=df)
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result.status == "success"
    assert result.data is not None
    assert result.data["constant"] == 0.0


def test_entropy_skips_all_null_column() -> None:
    """
    Ensure that a column with all missing values is excluded without error.
    """
    df = pd.DataFrame({"empty": [None] * 10})
    ctx, task = make_ctx_and_task(task_cls=ComputeEntropy, current_df=df)
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    assert result.status == "success"
    assert result.data is not None
    assert "empty" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_entropy_metadata_fields() -> None:
    """
    Check that metadata fields are populated with excluded_columns and column_types.
    """
    df = pd.DataFrame(
        {
            "col1": ["a", "b", "c", "a"],
            "col2": [1, 2, 3, 4],  # Should be excluded
        }
    )
    ctx, task = make_ctx_and_task(task_cls=ComputeEntropy, current_df=df)
    result: TaskResult = run_task_with_dependencies(ctx, task_cls=ComputeEntropy)

    metadata: dict[str, Any] = result.metadata
    assert "excluded_columns" in metadata
    assert "col2" in metadata["excluded_columns"]

    assert "column_types" in metadata
    assert "col1" in metadata["column_types"]
    assert "col2" in metadata["column_types"]
