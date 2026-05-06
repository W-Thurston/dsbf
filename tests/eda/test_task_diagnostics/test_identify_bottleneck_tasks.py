# tests/eda/test_task_diagnostics/test_identify_bottleneck_tasks.py

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas import DataFrame

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.identify_bottleneck_tasks import IdentifyBottleneckTasks
from tests.helpers.context_utils import make_ctx_and_task


@pytest.fixture
def base_df() -> DataFrame:
    return pd.DataFrame()  # dummy input


def test_top_n_bottlenecks_are_sorted(tmp_path, base_df) -> None:
    durations: dict[str, float] = {
        "A": 0.1,
        "B": 0.9,
        "C": 0.5,
        "D": 0.3,
        "E": 2.1,
    }

    ctx, task = make_ctx_and_task(
        task_cls=IdentifyBottleneckTasks,
        current_df=base_df,
        task_overrides={"top_n": 3},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.metadata["task_durations"] = durations

    result: TaskResult = ctx.run_task(task)

    top = result.summary["top_bottlenecks"]
    assert result.status == "success"
    assert len(top) == 3
    durations_sorted: list[float] = sorted(durations.values(), reverse=True)[:3]
    returned_durations: list = [entry["duration_sec"] for entry in top]
    assert returned_durations == [round(x, 4) for x in durations_sorted]


def test_handles_missing_task_durations(tmp_path, base_df) -> None:
    ctx, task = make_ctx_and_task(
        task_cls=IdentifyBottleneckTasks,
        current_df=base_df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)
    assert result.status == "failed"
    assert "No task durations" in result.summary["message"]


def test_handles_fewer_tasks_than_top_n(tmp_path, base_df) -> None:
    durations: dict[str, float] = {
        "A": 0.5,
        "B": 1.1,
    }

    ctx, task = make_ctx_and_task(
        task_cls=IdentifyBottleneckTasks,
        current_df=base_df,
        task_overrides={"top_n": 5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.metadata["task_durations"] = durations
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert len(result.summary["top_bottlenecks"]) == 2


def test_recommendation_triggers_for_slow_tasks(tmp_path, base_df) -> None:
    durations: dict[str, float] = {
        "train_big_model": 7.5,
        "fast_task": 0.1,
        "slow_loader": 6.2,
    }

    ctx, task = make_ctx_and_task(
        task_cls=IdentifyBottleneckTasks,
        current_df=base_df,
        task_overrides={"top_n": 3},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.metadata["task_durations"] = durations
    result: TaskResult = ctx.run_task(task)

    recs: list[str] | None = result.recommendations
    assert recs is not None
    assert len(recs) == 2
    assert any("train_big_model" in r for r in recs)
    assert all("took" in r for r in recs)


def test_output_format_is_stable(tmp_path, base_df) -> None:
    durations: dict[str, float] = {"task_x": 1.23456789}

    ctx, task = make_ctx_and_task(
        task_cls=IdentifyBottleneckTasks,
        current_df=base_df,
        task_overrides={"top_n": 1},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.metadata["task_durations"] = durations
    result: TaskResult = ctx.run_task(task)

    top = result.summary["top_bottlenecks"]
    assert len(top) == 1
    assert isinstance(top[0]["duration_sec"], float)
    assert round(top[0]["duration_sec"], 4) == 1.2346


def test_bottleneck_plot_generated(tmp_path, base_df) -> None:
    durations: dict[str, float] = {
        "slow_loader": 6.0,
        "big_model": 9.2,
        "prep": 3.5,
        "fast_task": 0.1,
    }

    ctx, task = make_ctx_and_task(
        task_cls=IdentifyBottleneckTasks,
        current_df=base_df,
        task_overrides={"top_n": 3},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.metadata["task_durations"] = durations
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    # Task produces summary findings; static plot generation was removed.
    # Verify the top bottlenecks are present in the summary instead.
    assert "top_bottlenecks" in result.summary
    assert len(result.summary["top_bottlenecks"]) == 3

    plot_entry: dict[str, Any] = result.plots["bottleneck_tasks"]
    static_path = plot_entry["static"]
    interactive = plot_entry["interactive"]

    assert isinstance(static_path, Path)
    assert static_path.exists()
    assert static_path.suffix == ".png"

    assert interactive["type"] == "bar"
    assert all(re.match(r".+:\s*\d+(\.\d+)?s$", a) for a in interactive["annotations"])
