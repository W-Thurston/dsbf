# tests/eda/test_tasks/test_detect_skewness.py

import warnings
from pathlib import Path

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_skewness import DetectSkewness
from tests.helpers.context_utils import make_ctx_and_task


def test_detect_skewness_expected_output(tmp_path):
    warnings.filterwarnings(
        "ignore", category=PendingDeprecationWarning, module="seaborn"
    )
    df = pd.DataFrame(
        {
            "normal": [1, 2, 3, 4, 5],
            "skewed": [1, 1, 1, 2, 100],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert "normal" in result.data
    assert "skewed" in result.data
    assert abs(result.data["normal"]) < 1.0
    assert result.data["skewed"] > 1.0

    # Plot checks
    assert result.plots is not None
    assert "skewed" in result.plots
    static_path = result.plots["skewed"]["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
    static_path.unlink()

    interactive = result.plots["skewed"]["interactive"]
    assert isinstance(interactive, dict)
    assert "annotations" in interactive
    assert any("Skewness:" in a for a in interactive["annotations"])


def test_detect_skewness_all_nulls(tmp_path):
    df = pd.DataFrame({"a": [None, None, None], "b": [float("nan")] * 3})

    ctx, task = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)
    assert result.status == "success"
    assert result.data == {}
    assert result.plots == {}


def test_detect_skewness_constant_column(tmp_path):
    df = pd.DataFrame({"const": [5, 5, 5, 5, 5]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)
    assert result.status == "success"
    assert result.data is not None
    assert result.data["const"] == 0.0 or abs(result.data["const"]) < 1e-9
    assert result.plots is not None
    assert "const" in result.plots
    static_path = result.plots["const"]["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
    static_path.unlink()
