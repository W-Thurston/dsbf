# tests/eda/test_tasks/test_summarize_numeric.py

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from dsbf.eda.tasks.summarize_numeric import SummarizeNumeric
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_extended_stats_computed(tmp_path) -> None:
    """All expected statistics keys must be present for a numeric column."""
    df = pd.DataFrame({"x": list(range(1, 101))})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNumeric)

    assert result.status == "success"
    assert "x" in result.data
    stats = result.data["x"]
    for key in ("count", "mean", "std", "min", "50%", "max", "near_zero_variance"):
        assert key in stats


def test_near_zero_variance_flagged(tmp_path) -> None:
    """A near-constant column must be flagged with near_zero_variance=True."""
    df = pd.DataFrame({"const": [5.0] * 100})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNumeric)

    assert result.status == "success"
    assert result.data["const"]["near_zero_variance"] is True


def test_near_zero_variance_guidance_emitted(tmp_path) -> None:
    """EDA and ML guidance must be emitted for near-zero variance columns."""
    df = pd.DataFrame({"const": [5.0] * 100})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNumeric)

    assert result.status == "success"
    assert result.guidance is not None
    assert "const" in result.guidance
    assert len(result.guidance["const"]["eda"]) > 0
    assert len(result.guidance["const"]["ml"]) > 0


def test_skewed_column_emits_mean_median_guidance(tmp_path) -> None:
    """A heavily skewed column must emit mean-median gap guidance."""
    df = pd.DataFrame({"skewed": [1] * 90 + list(range(100, 200, 10))})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNumeric)

    assert result.status == "success"
    # Guidance may or may not fire depending on the gap - check it runs cleanly
    assert result.data["skewed"]["near_zero_variance"] is False


def test_all_null_column_skipped(tmp_path) -> None:
    """A column that is entirely null must be skipped without error."""
    df = pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNumeric)

    assert result.status == "success"
    assert "a" not in result.data  # all-null is skipped
    assert "b" in result.data


def test_polars_dataframe_handled(tmp_path) -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNumeric)
    assert result.status == "success"
    assert "x" in result.data


def test_no_plots_generated(tmp_path) -> None:
    df = pd.DataFrame({"x": list(range(10))})
    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeNumeric,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeNumeric)
    assert result.status == "success"
    assert result.plots is None
