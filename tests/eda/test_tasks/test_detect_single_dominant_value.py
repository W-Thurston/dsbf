# tests/eda/test_tasks/test_detect_single_dominant_value.py

from pathlib import Path

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_single_dominant_value import DetectSingleDominantValue
from tests.helpers.context_utils import make_ctx_and_task


def test_detect_single_dominant_value_expected_output(tmp_path):
    df = pd.DataFrame(
        {
            "mostly_ones": [1] * 95 + [0] * 5,
            "uniform": [1, 2, 3, 4, 5] * 20,
            "binary": ["yes"] * 96 + ["no"] * 4,
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectSingleDominantValue,
        current_df=df,
        task_overrides={"dominance_threshold": 0.9},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "mostly_ones" in result.data
    assert "binary" in result.data
    assert "uniform" not in result.data


def test_detect_single_dominant_value_with_plots(tmp_path):
    df = pd.DataFrame(
        {
            "fruit": ["apple"] * 96 + ["banana"] * 4,
            "color": ["red", "green", "blue", "red", "green"] * 20,
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectSingleDominantValue,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
        task_overrides={"dominance_threshold": 0.9},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "fruit" in result.plots
    assert "color" not in result.plots

    static_path = result.plots["fruit"]["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
    static_path.unlink()

    interactive = result.plots["fruit"]["interactive"]
    assert "annotations" in interactive
    assert any("Dominant value" in a for a in interactive["annotations"])


def test_detect_single_dominant_value_all_null(tmp_path):
    df = pd.DataFrame({"col": [None, None, None]})
    ctx, task = make_ctx_and_task(
        task_cls=DetectSingleDominantValue,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)
    assert result.status == "success"
    assert result.data == {}
    assert result.plots == {}


def test_detect_single_dominant_value_constant_column(tmp_path):
    df = pd.DataFrame({"col": ["A"] * 100})
    ctx, task = make_ctx_and_task(
        task_cls=DetectSingleDominantValue,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)
    assert result.status == "success"
    assert result.data is not None
    assert result.data.get("col") is not None
    assert result.plots is not None
    assert "col" in result.plots
    assert result.plots["col"]["static"].exists()
    result.plots["col"]["static"].unlink()
