# tests/eda/test_tasks/test_detect_skewness.py

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from dsbf.eda.tasks.detect_skewness import DetectSkewness
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_skewness_computed_for_numeric_columns(tmp_path) -> None:
    """Skewness must be computed for all continuous columns."""
    df = pd.DataFrame(
        {
            "normal": [1, 2, 3, 4, 5],
            "skewed": [1, 1, 1, 2, 100],
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectSkewness)

    assert result.status == "success"
    assert result.data is not None
    assert "normal" in result.data
    assert "skewed" in result.data
    assert abs(result.data["normal"]) < 1.0
    assert result.data["skewed"] > 1.0


def test_guidance_attached_for_skewed_columns(tmp_path) -> None:
    """EDA and ML guidance blurbs must be attached for columns with notable skew."""
    df = pd.DataFrame({"heavily_skewed": [1] * 90 + list(range(100, 200, 10))})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectSkewness)

    assert result.status == "success"
    assert result.guidance is not None
    assert "heavily_skewed" in result.guidance
    assert len(result.guidance["heavily_skewed"]["eda"]) > 0
    assert len(result.guidance["heavily_skewed"]["ml"]) > 0
    ml_actions = result.guidance["heavily_skewed"]["ml"][0]["actions"]
    assert any(a["action"] == "transform" for a in ml_actions)


def test_symmetric_column_produces_no_guidance(tmp_path) -> None:
    """A symmetric column (|skew| ≤ 0.5) must not emit any guidance blurbs."""
    df = pd.DataFrame({"symmetric": list(range(1, 11))})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectSkewness)

    assert result.status == "success"
    # No guidance for symmetric columns
    assert result.guidance is None or "symmetric" not in (result.guidance or {})


def test_all_null_columns_produce_empty_data(tmp_path) -> None:
    """Columns that are entirely null must be skipped and produce empty data."""
    df = pd.DataFrame({"a": [None, None, None], "b": [float("nan")] * 3})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectSkewness)

    assert result.status == "success"
    assert result.data == {}
    assert result.plots is None


def test_constant_column_skewness_is_zero(tmp_path) -> None:
    """A constant column must have skewness 0.0."""
    df = pd.DataFrame({"const": [5, 5, 5, 5, 5]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectSkewness)

    assert result.status == "success"
    assert "const" in result.data
    assert abs(result.data["const"]) < 1e-9


def test_metadata_column_types_populated(tmp_path) -> None:
    """column_types metadata must include inferred and intent dtypes for all columns."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectSkewness)

    assert result.status == "success"
    column_types = result.metadata.get("column_types", {})
    assert "a" in column_types
    assert column_types["a"]["analysis_intent_dtype"] == "continuous"


def test_no_plots_generated(tmp_path) -> None:
    """Skewness task must not generate plots."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 100]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectSkewness)

    assert result.status == "success"
    assert result.plots is None
