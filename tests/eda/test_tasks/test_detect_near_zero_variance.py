# tests/eda/test_tasks/test_detect_near_zero_variance.py

import warnings
from pathlib import Path

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_near_zero_variance import DetectNearZeroVariance
from tests.helpers.context_utils import make_ctx_and_task


def test_detects_near_zero_variance(tmp_path):
    warnings.filterwarnings(
        "ignore", category=PendingDeprecationWarning, module="seaborn"
    )
    df = pl.DataFrame(
        {
            "constant": [3.14] * 100,
            "low_var": [1.00001 + (i % 2) * 0.00001 for i in range(100)],
            "normal": list(range(100)),
        }
    )
    ctx, task = make_ctx_and_task(
        task_cls=DetectNearZeroVariance,
        current_df=df,
        task_overrides={"threshold": 1e-4},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "constant" in result.data["low_variance_columns"]
    assert "low_var" in result.data["low_variance_columns"]
    assert "normal" not in result.data["low_variance_columns"]


def test_skips_non_numeric(tmp_path):
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "category": ["yes", "no", "yes", "no"],
        }
    )
    ctx, task = make_ctx_and_task(
        task_cls=DetectNearZeroVariance,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.data == {"low_variance_columns": {}}
    assert result.recommendations is not None
    assert len(result.recommendations) == 0


def test_detect_near_zero_variance_with_plots(tmp_path):
    warnings.filterwarnings(
        "ignore", category=PendingDeprecationWarning, module="seaborn"
    )
    df = pd.DataFrame(
        {
            "const": [5] * 10,
            "low": [
                1.001,
                1.0001,
                1.0002,
                1.001,
                1.0001,
                1.0002,
                1.001,
                1.0001,
                1.0002,
                1.001,
            ],
            "normal": list(range(10)),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectNearZeroVariance,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
        task_overrides={"threshold": 1e-3},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "const" in result.plots
    assert "low" in result.plots
    assert "normal" not in result.plots

    # Check static file exists and delete
    for col in ["const", "low"]:
        static_path = result.plots[col]["static"]
        assert isinstance(static_path, Path)
        assert static_path.exists()
        static_path.unlink()

        # Confirm interactive annotation
        interactive = result.plots[col]["interactive"]
        assert isinstance(interactive, dict)
        assert "annotations" in interactive
        assert any("Variance" in a for a in interactive["annotations"])
