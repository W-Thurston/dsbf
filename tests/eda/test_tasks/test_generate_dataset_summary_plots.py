# tests/eda/test_tasks/test_generate_dataset_summary_plots.py

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from dsbf.eda.tasks.generate_dataset_summary_plots import GenerateDatasetSummaryPlots
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def _assert_plot_artifact(value: Any) -> None:
    """Validate a static or interactive plot artifact reference."""
    if isinstance(value, str):
        assert Path(value).exists(), f"Static artifact path does not exist: {value}"
    elif isinstance(value, dict):
        return  # interactive artifacts are dicts; content varies by PlotFactory
    else:
        msg: str = f"Unexpected artifact type: {type(value)}"
        raise AssertionError(msg)  # noqa: TRY004


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_typical_dataset_produces_all_expected_plots(tmp_path) -> None:
    """A typical dataset with numeric and categorical columns produces all plots."""
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3, None],
            "num2": [0.1, 0.2, 0.3, 0.4],
            "cat": ["a", "b", "a", "c"],
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=GenerateDatasetSummaryPlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    # Inject metadata directly - GenerateDatasetSummaryPlots reads context metadata
    # for the dtype stacked bar, and run_task_with_dependencies runs infer_types first.
    result: TaskResult = run_task_with_dependencies(ctx, GenerateDatasetSummaryPlots)

    assert result.status == "success"
    assert result.data is not None

    expected: set[str] = {"correlation_matrix", "null_matrix", "missingness_matrix"}
    assert expected.issubset(result.data)

    for artifacts in result.data.values():
        for key in ("static", "interactive"):
            if key in artifacts:
                _assert_plot_artifact(artifacts[key])


def test_no_numeric_columns_skips_correlation(tmp_path) -> None:
    """A dataset with no numeric columns must skip the correlation matrix."""
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})

    ctx, _ = make_ctx_and_task(
        task_cls=GenerateDatasetSummaryPlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, GenerateDatasetSummaryPlots)

    assert result.status == "success"
    # correlation_matrix should be empty dict when < 2 numeric columns
    assert result.data.get("correlation_matrix", {}) == {}


def test_single_numeric_column_skips_correlation(tmp_path) -> None:
    """Fewer than 2 numeric columns must produce an empty correlation_matrix entry."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    ctx, _ = make_ctx_and_task(
        task_cls=GenerateDatasetSummaryPlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, GenerateDatasetSummaryPlots)

    assert result.status == "success"
    assert result.data.get("correlation_matrix", {}) == {}


def test_no_column_plots_generated(tmp_path) -> None:
    """Task must not populate result.plots - it stores artifacts in result.data."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    ctx, _ = make_ctx_and_task(
        task_cls=GenerateDatasetSummaryPlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, GenerateDatasetSummaryPlots)

    assert result.status == "success"
    assert result.plots is None
