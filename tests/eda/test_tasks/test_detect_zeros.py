# tests/eda/test_tasks/test_detect_zeros.py

from pathlib import Path

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_zeros import DetectZeros
from tests.helpers.context_utils import make_ctx_and_task


def test_detect_zeros_expected_output(tmp_path):
    df = pd.DataFrame(
        {
            "a": [0, 0, 1, 2, 3, 0, 4, 0, 5, 0],  # 5 zeros -> 50%
            "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 0 zeros
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectZeros,
        current_df=df,
        task_overrides={"flag_threshold": 0.3},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    counts = result.data["zero_counts"]
    flags = result.data["zero_flags"]
    percentages = result.data["zero_percentages"]

    assert counts["a"] == 5
    assert flags["a"] is True
    assert percentages["a"] == 0.5
    assert flags["b"] is False


def test_detect_zeros_all_zeros_or_none(tmp_path):
    df = pd.DataFrame({"a": [0, 0, 0, 0], "b": [1, 2, 3, 4]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectZeros,
        current_df=df,
        task_overrides={"flag_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.data["zero_flags"]["a"] is True
    assert result.data["zero_flags"]["b"] is False


def test_detect_zeros_generates_plot(tmp_path):
    """
    Test that DetectZeros generates a barplot when any zeros are found.
    """
    df = pd.DataFrame(
        {
            "col1": [0, 0, 0, 1],  # 75% zeros
            "col2": [1, 2, 3, 4],  # 0% zeros
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectZeros,
        current_df=df,
        task_overrides={"flag_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "zero_percentages" in result.plots

    plot_entry = result.plots["zero_percentages"]

    # Check static plot path
    static_path = plot_entry["static"]
    assert isinstance(static_path, Path)
    assert static_path.suffix == ".png"
    assert static_path.exists()

    # Check interactive format
    interactive = plot_entry["interactive"]
    assert isinstance(interactive, dict)
    assert interactive["type"] == "bar"
    assert "data" in interactive
    assert "config" in interactive
    assert "annotations" in interactive
    assert "zero" in interactive["config"]["title"].lower()


def test_detect_zeros_skips_plot_if_no_zeros(tmp_path):
    """
    Test that DetectZeros does not produce a plot when no zeros are present.
    """
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [5, 6, 7, 8],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectZeros,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.plots is None or result.plots == {}
