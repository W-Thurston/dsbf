# tests/eda/test_tasks/test_detect_high_cardinality.py

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_high_cardinality import DetectHighCardinality
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


def test_high_cardinality_column_detected(tmp_path) -> None:
    """A column with more unique values than the threshold must be flagged."""
    df = pd.DataFrame(
        {
            # 100 unique string values
            # will be classified as categorical by infer_types
            "city": [f"city_{i}" for i in range(100)],
            # 5 unique values - below threshold
            "region": list("ABCDE") * 20,
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectHighCardinality,
        current_df=df,
        task_overrides={"cardinality_threshold": 50},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"city": "categorical", "region": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, DetectHighCardinality)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    # city has 100 unique values > threshold 50
    assert "city" in result.data
    # region has only 5 unique values
    assert "region" not in result.data
    assert result.metadata["cardinality_threshold"] == 50


def test_low_cardinality_column_not_flagged(tmp_path) -> None:
    """A column below the threshold must not appear in results."""
    df = pd.DataFrame({"label": ["x", "x", "y", "z", "x", "y", "z"]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectHighCardinality,
        current_df=df,
        task_overrides={"cardinality_threshold": 10},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectHighCardinality)

    assert result.status == "success"
    assert "label" not in result.data


def test_guidance_attached_for_flagged_columns(tmp_path) -> None:
    """EDA and ML guidance blurbs must be attached for each high-cardinality column."""
    df = pd.DataFrame({"sku": [f"SKU-{i}" for i in range(100)]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectHighCardinality,
        current_df=df,
        task_overrides={"cardinality_threshold": 50},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"sku": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, DetectHighCardinality)

    assert result.status == "success"
    assert "sku" in result.data
    assert result.guidance is not None
    assert "sku" in result.guidance
    assert len(result.guidance["sku"]["eda"]) > 0
    assert len(result.guidance["sku"]["ml"]) > 0
    ml_actions = result.guidance["sku"]["ml"][0]["actions"]
    assert any(a["action"] == "encode" for a in ml_actions)


def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must work correctly on Polars DataFrames."""
    df = pl.DataFrame({"cat": [f"val_{i}" for i in range(100)]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectHighCardinality,
        current_df=df,
        task_overrides={"cardinality_threshold": 50},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"cat": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, DetectHighCardinality)

    assert result.status == "success"
    assert "cat" in result.data
    assert result.data["cat"] == 100


def test_all_null_column_not_flagged(tmp_path) -> None:
    """A column with all null values must not appear in results."""
    df = pd.DataFrame({"x": [None, None, None]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectHighCardinality,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectHighCardinality)

    assert result.status == "success"
    assert result.data == {}


def test_no_plots_generated(tmp_path) -> None:
    """High-cardinality detection task must not generate plots."""
    df = pd.DataFrame({"tag": [f"t{i}" for i in range(100)]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectHighCardinality,
        current_df=df,
        task_overrides={"cardinality_threshold": 50},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectHighCardinality)

    assert result.status == "success"
    assert result.plots is None
