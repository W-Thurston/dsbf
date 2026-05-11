# tests/eda/test_tasks/test_detect_near_zero_variance.py

import pandas as pd
import polars as pl
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_near_zero_variance import DetectNearZeroVariance
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_detects_near_zero_variance_columns(tmp_path) -> None:
    """Constant and near-constant columns must be flagged; varying columns must not."""
    df = pl.DataFrame(
        {
            "constant": [3.14] * 100,
            "low_var": [1.00001 + (i % 2) * 0.00001 for i in range(100)],
            "normal": list(range(100)),
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectNearZeroVariance,
        current_df=df,
        task_overrides={"threshold": 1e-4},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectNearZeroVariance)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "constant" in result.data["low_variance_columns"]
    assert "low_var" in result.data["low_variance_columns"]
    assert "normal" not in result.data["low_variance_columns"]


def test_skips_non_numeric_columns(tmp_path) -> None:
    """Categorical-only datasets must produce an empty low_variance_columns dict."""
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "category": ["yes", "no", "yes", "no"],
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectNearZeroVariance,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectNearZeroVariance)

    assert result.status == "success"
    # infer_types classifies "id" and "category" as non-continuous.
    # get_columns_by_intent returns 0 eligible cols → make_empty_result fires.
    assert result.data == {}
    assert result.recommendations is None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_guidance_attached_for_flagged_columns(tmp_path) -> None:
    """EDA and ML guidance blurbs must be attached for each low-variance column."""
    df = pd.DataFrame({"const": [5.0] * 50, "varying": list(range(50))})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectNearZeroVariance,
        current_df=df,
        task_overrides={"threshold": 1e-4},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectNearZeroVariance)

    assert result.status == "success"
    assert "const" in result.data["low_variance_columns"]
    assert result.guidance is not None
    assert "const" in result.guidance
    assert len(result.guidance["const"]["eda"]) > 0
    assert len(result.guidance["const"]["ml"]) > 0
    assert result.guidance["const"]["ml"][0]["actions"][0]["action"] == "drop"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_no_plots_generated(tmp_path) -> None:
    """Near-zero variance task must not generate plots."""
    df = pl.DataFrame({"const": [1.0] * 100, "vary": list(range(100))})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectNearZeroVariance,
        current_df=df,
        task_overrides={"threshold": 1e-4},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectNearZeroVariance)

    assert result.status == "success"
    assert result.plots is None


def test_threshold_configurable(tmp_path) -> None:
    """A tighter threshold must flag fewer columns than a looser one."""
    df = pd.DataFrame(
        {
            "mid_var": [1.0 + i * 0.001 for i in range(100)],  # var ≈ 0.00083
            "low_var": [1.0 + i * 0.0001 for i in range(100)],  # var much smaller
        },
    )

    ctx_tight, _ = make_ctx_and_task(
        task_cls=DetectNearZeroVariance,
        current_df=df,
        task_overrides={"threshold": 1e-6},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result_tight: TaskResult = run_task_with_dependencies(
        ctx_tight, DetectNearZeroVariance
    )

    ctx_loose, _ = make_ctx_and_task(
        task_cls=DetectNearZeroVariance,
        current_df=df,
        task_overrides={"threshold": 1e-2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result_loose: TaskResult = run_task_with_dependencies(
        ctx_loose, DetectNearZeroVariance
    )

    tight_count: int = len(result_tight.data["low_variance_columns"])
    loose_count: int = len(result_loose.data["low_variance_columns"])
    assert loose_count >= tight_count
