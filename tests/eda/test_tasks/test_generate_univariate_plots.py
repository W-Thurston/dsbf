# tests/eda/test_tasks/test_generate_univariate_plots.py

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.generate_univariate_plots import GenerateUnivariatePlots
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def _assert_static_artifact(value: Any) -> None:
    """Validate that a static artifact path exists on disk."""
    if isinstance(value, str):
        assert Path(value).exists(), f"Static artifact does not exist: {value}"
    elif isinstance(value, dict):
        for v in value.values():
            _assert_static_artifact(v)
    else:
        msg: str = f"Unexpected static artifact type: {type(value)}"
        raise AssertionError(msg)  # noqa: TRY004


def _assert_interactive_artifact(value: Any) -> None:
    """Validate that an interactive artifact is a string path or dict."""
    if isinstance(value, str | dict):
        return
    msg: str = f"Unexpected interactive artifact type: {type(value)}"
    raise AssertionError(msg)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_typical_dataset_plots_categorical_and_continuous(tmp_path) -> None:
    """Categorical and continuous columns must produce appropriate plot entries."""
    df = pd.DataFrame(
        {
            "category": ["A", "B", "A"],
            "numeric": [1, 2, 3],
            "binary": [0, 1, 0],  # nunique=2 → categorical
            "empty": [None, None, None],  # unknown → skipped
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=GenerateUnivariatePlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, GenerateUnivariatePlots)

    assert result.status == "success"
    assert result.data is not None

    # empty column (all-null → unknown) must be skipped
    assert "empty" not in result.data
    # categorical and numeric columns must appear
    assert "category" in result.data
    assert "numeric" in result.data

    for plots in result.data.values():
        for artifacts in plots.values():
            if "static" in artifacts:
                _assert_static_artifact(artifacts["static"])
            if "interactive" in artifacts:
                _assert_interactive_artifact(artifacts["interactive"])


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_categorical_column_produces_bar_plot(tmp_path) -> None:
    """A categorical column must produce a 'bar' entry in its plot results."""
    df = pd.DataFrame({"label": ["A", "B", "A", "C", "B"]})

    ctx, _ = make_ctx_and_task(
        task_cls=GenerateUnivariatePlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, GenerateUnivariatePlots)

    assert result.status == "success"
    assert "label" in result.data
    assert "bar" in result.data["label"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_continuous_column_produces_histogram_and_boxplot(tmp_path) -> None:
    """A continuous column must produce histogram and boxplot entries."""
    df = pd.DataFrame({"value": list(range(20))})

    ctx, _ = make_ctx_and_task(
        task_cls=GenerateUnivariatePlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, GenerateUnivariatePlots)

    assert result.status == "success"
    assert "value" in result.data
    assert "histogram" in result.data["value"]
    assert "boxplot" in result.data["value"]


def test_all_null_column_skipped(tmp_path) -> None:
    """A column that is entirely null must be skipped without error."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})

    ctx, _ = make_ctx_and_task(
        task_cls=GenerateUnivariatePlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, GenerateUnivariatePlots)

    assert result.status == "success"
    assert "b" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must work correctly on Polars DataFrames."""
    df = pl.DataFrame({"cat": ["A", "B", "A"], "num": [1.0, 2.0, 3.0]})

    ctx, _ = make_ctx_and_task(
        task_cls=GenerateUnivariatePlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, GenerateUnivariatePlots)

    assert result.status == "success"
    assert result.data is not None
    assert len(result.data) > 0


def test_no_column_plots_field(tmp_path) -> None:
    """result.plots must be None - plot artifacts are stored in result.data."""
    df = pd.DataFrame({"x": [1, 2, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=GenerateUnivariatePlots,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, GenerateUnivariatePlots)

    assert result.status == "success"
    assert result.plots is None
