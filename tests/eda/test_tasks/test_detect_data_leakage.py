# tests/eda/test_tasks/test_detect_data_leakage.py

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_data_leakage import DetectDataLeakage
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


def test_leakage_pair_detected(tmp_path) -> None:
    """Perfectly correlated column pair must appear in leakage_pairs."""
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],  # perfect Pearson with x
            "z": [5, 4, 3, 2, 1],
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDataLeakage,
        current_df=df,
        task_overrides={"correlation_threshold": 0.99},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDataLeakage)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    leakage = result.data["leakage_pairs"]
    assert isinstance(leakage, dict)
    assert any("x|y" in key or "y|x" in key for key in leakage)


def test_no_leakage_on_uncorrelated_data(tmp_path) -> None:
    """Uncorrelated columns must produce no leakage pairs."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 1, 4, 2, 3],  # low correlation
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDataLeakage,
        current_df=df,
        task_overrides={"correlation_threshold": 0.99},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDataLeakage)

    assert result.status == "success"
    assert result.data["leakage_pairs"] == {}


def test_guidance_attached_for_both_columns(tmp_path) -> None:
    """Both columns in a leakage pair must receive EDA and ML guidance blurbs."""
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDataLeakage,
        current_df=df,
        task_overrides={"correlation_threshold": 0.99},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDataLeakage)

    assert result.status == "success"
    assert result.guidance is not None

    # Both x and y must have guidance
    assert "x" in result.guidance
    assert "y" in result.guidance

    for col in ("x", "y"):
        assert len(result.guidance[col]["eda"]) > 0
        assert len(result.guidance[col]["ml"]) > 0
        assert result.guidance[col]["eda"][0]["level"] == "error"
        ml_actions = result.guidance[col]["ml"][0]["actions"]
        assert any(a["action"] == "drop" for a in ml_actions)


def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames by converting to pandas internally."""
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],  # perfect correlation
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDataLeakage,
        current_df=df,
        task_overrides={"correlation_threshold": 0.99},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDataLeakage)

    assert result.status == "success"
    assert "a|b" in result.data["leakage_pairs"]


def test_no_plots_generated(tmp_path) -> None:
    """Leakage detection task must not generate plots."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDataLeakage,
        current_df=df,
        task_overrides={"correlation_threshold": 0.99},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDataLeakage)

    assert result.status == "success"
    assert result.plots is None


def test_metadata_threshold_stored(tmp_path) -> None:
    """Configured correlation_threshold must be stored in metadata."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectDataLeakage,
        current_df=df,
        task_overrides={"correlation_threshold": 0.95},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectDataLeakage)

    assert result.metadata["correlation_threshold"] == 0.95
