# tests/eda/test_tasks/test_compute_pairwise_associations.py

from typing import TYPE_CHECKING, Literal

import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.compute_pairwise_associations import ComputePairwiseAssociations
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_pearson_r_for_continuous_pair(tmp_path) -> None:
    """Pearson r must be computed for two continuous columns."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "y": [2.0, 4.0, 6.0, 8.0, 10.0] * 10,  # perfect correlation
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    assert result.data is not None
    assert "x|y" in result.data
    entry = result.data["x|y"]
    assert entry["metric_type"] == "pearson_r"
    assert abs(entry["metric"] - 1.0) < 1e-4
    assert entry["strength"] == "strong"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_cramers_v_for_categorical_pair(tmp_path) -> None:
    """Cramér's V must be computed for two categorical columns."""
    df = pd.DataFrame(
        {
            "color": ["red", "red", "blue", "blue", "green"] * 10,
            "shape": ["circle", "circle", "square", "square", "triangle"] * 10,
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    assert "color|shape" in result.data
    entry = result.data["color|shape"]
    assert entry["metric_type"] == "cramers_v"
    assert 0.0 <= entry["metric"] <= 1.0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_eta_squared_for_continuous_categorical_pair(tmp_path) -> None:
    """Eta squared must be computed for continuous x multi-level categorical pairs."""
    df = pd.DataFrame(
        {
            "score": [10.0, 20.0, 30.0, 40.0, 50.0] * 10,
            "group": ["A", "A", "B", "B", "C"] * 10,
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    key: Literal["group|score", "score|group"] = (
        "score|group" if "score|group" in result.data else "group|score"
    )
    assert key in result.data
    assert result.data[key]["metric_type"] == "eta_squared"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_id_and_datetime_columns_skipped(tmp_path) -> None:
    """Columns typed as id or datetime must not appear in associations."""
    df = pd.DataFrame(
        {
            "id_col": range(50),
            "value": range(50),
            "label": ["a", "b"] * 25,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    # Inject semantic types directly — infer_types classifies range(50) integers
    # as continuous, not id. We inject explicitly to test the task's skip logic
    # in isolation from the type inference decision.
    ctx.set_metadata(
        "semantic_types",
        {
            "id_col": "id",
            "value": "continuous",
            "label": "categorical",
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    for key in result.data:
        col_a, col_b = key.split("|")
        assert col_a != "id_col"
        assert col_b != "id_col"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_strength_labels_present(tmp_path) -> None:
    """Every association entry must have a strength label."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "b": [5.0, 4.0, 3.0, 2.0, 1.0] * 10,
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    valid_strengths: set[str] = {"strong", "moderate", "weak", "negligible"}
    for entry in result.data.values():
        assert entry["strength"] in valid_strengths


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_match_data(tmp_path) -> None:
    """Summary pair_count and strength_counts must be consistent with data."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "y": [2.0, 4.0, 6.0, 8.0, 10.0] * 10,
            "z": [5.0, 4.0, 3.0, 2.0, 1.0] * 10,
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    assert result.summary["pair_count"] == len(result.data)
    total_by_strength = sum(result.summary["strength_counts"].values())
    assert total_by_strength == len(result.data)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames by converting to pandas internally."""
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "b": [2.0, 4.0, 6.0, 8.0, 10.0] * 10,
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    assert result.data is not None
    assert len(result.data) > 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_reliability_warning_on_low_n(tmp_path) -> None:
    """A strong_warning must be emitted when N < 30."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"min_sample_size": 2},  # lower min so pairs aren't skipped
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    assert result.reliability_warnings is not None
    assert "low_row_count" in result.reliability_warnings.get("strong_warning", {})


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_no_plots_generated(tmp_path) -> None:
    """Pairwise associations task must not generate plots — rendering is
    handled by the Relationships tab on demand."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "b": [2.0, 4.0, 6.0, 8.0, 10.0] * 10,
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    assert result.plots is None
