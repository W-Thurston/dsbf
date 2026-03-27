# tests/eda/test_tasks/test_detect_id_columns.py

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_id_columns import DetectIdColumns
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


def test_string_id_columns_detected(tmp_path) -> None:
    """String columns with near-unique values must be flagged as likely IDs."""
    df = pd.DataFrame(
        {
            # 100 unique strings - near-100% uniqueness → id
            "uuid": [f"user_{i}" for i in range(100)],
            # 100 unique strings - also id
            "order_id": [f"ord_{i}" for i in range(100)],
            # repeating values - not an id
            "status": ["active", "inactive"] * 50,
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectIdColumns,
        current_df=df,
        task_overrides={"threshold_ratio": 0.95},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectIdColumns)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "uuid" in result.data
    assert "order_id" in result.data
    assert "status" not in result.data
    assert result.metadata["threshold_ratio"] == 0.95


def test_no_id_columns_in_low_cardinality_dataset(tmp_path) -> None:
    """A dataset with no near-unique columns must return empty results."""
    df = pd.DataFrame(
        {
            "group": ["A", "B", "A", "B"],
            "value": [10, 10, 20, 20],
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectIdColumns,
        current_df=df,
        task_overrides={"threshold_ratio": 0.95},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectIdColumns)

    assert result.status == "success"
    assert result.data == {}


def test_guidance_attached_for_id_columns(tmp_path) -> None:
    """EDA and ML guidance blurbs must be attached for detected ID columns."""
    df = pd.DataFrame({"user_id": [f"u{i}" for i in range(100)]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectIdColumns,
        current_df=df,
        task_overrides={"threshold_ratio": 0.95},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectIdColumns)

    assert result.status == "success"
    assert "user_id" in result.data
    assert result.guidance is not None
    assert "user_id" in result.guidance
    assert len(result.guidance["user_id"]["eda"]) > 0
    assert len(result.guidance["user_id"]["ml"]) > 0
    ml_actions = result.guidance["user_id"]["ml"][0]["actions"]
    assert any(a["action"] == "drop" for a in ml_actions)


def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must work correctly on Polars DataFrames."""
    df = pl.DataFrame(
        {
            "record_id": [f"rec_{i}" for i in range(100)],
            "category": ["A", "B"] * 50,
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectIdColumns,
        current_df=df,
        task_overrides={"threshold_ratio": 0.95},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectIdColumns)

    assert result.status == "success"
    assert "record_id" in result.data
    assert "category" not in result.data


def test_no_plots_generated(tmp_path) -> None:
    """ID column detection must not generate plots."""
    df = pd.DataFrame({"id_col": [f"x{i}" for i in range(100)]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectIdColumns,
        current_df=df,
        task_overrides={"threshold_ratio": 0.95},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectIdColumns)

    assert result.status == "success"
    assert result.plots is None
