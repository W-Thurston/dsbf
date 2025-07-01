# tests/eda/test_tasks/test_identify_bottleneck_tasks.py

import pandas as pd
import pytest

from dsbf.eda.tasks.identify_bottleneck_tasks import IdentifyBottleneckTasks
from tests.helpers.context_utils import make_ctx_and_task


@pytest.fixture
def base_df():
    return pd.DataFrame()  # dummy input


def test_top_n_bottlenecks_are_sorted(base_df):
    durations = {
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
    )
    ctx.metadata["task_durations"] = durations

    result = ctx.run_task(task)

    top = result.summary["top_bottlenecks"]
    assert result.status == "success"
    assert len(top) == 3
    durations_sorted = sorted(durations.values(), reverse=True)[:3]
    returned_durations = [entry["duration_sec"] for entry in top]
    assert returned_durations == [round(x, 4) for x in durations_sorted]


def test_handles_missing_task_durations(base_df):
    ctx, task = make_ctx_and_task(
        task_cls=IdentifyBottleneckTasks,
        current_df=base_df,
    )
    result = ctx.run_task(task)
    assert result.status == "failed"
    assert "No task durations" in result.summary["message"]


def test_handles_fewer_tasks_than_top_n(base_df):
    durations = {
        "A": 0.5,
        "B": 1.1,
    }

    ctx, task = make_ctx_and_task(
        task_cls=IdentifyBottleneckTasks,
        current_df=base_df,
        task_overrides={"top_n": 5},
    )
    ctx.metadata["task_durations"] = durations
    result = ctx.run_task(task)

    assert result.status == "success"
    assert len(result.summary["top_bottlenecks"]) == 2


def test_recommendation_triggers_for_slow_tasks(base_df):
    durations = {
        "train_big_model": 7.5,
        "fast_task": 0.1,
        "slow_loader": 6.2,
    }

    ctx, task = make_ctx_and_task(
        task_cls=IdentifyBottleneckTasks,
        current_df=base_df,
        task_overrides={"top_n": 3},
    )
    ctx.metadata["task_durations"] = durations
    result = ctx.run_task(task)

    recs = result.recommendations
    assert recs is not None
    assert len(recs) == 2
    assert any("train_big_model" in r for r in recs)
    assert all("took" in r for r in recs)


def test_output_format_is_stable(base_df):
    durations = {"task_x": 1.23456789}

    ctx, task = make_ctx_and_task(
        task_cls=IdentifyBottleneckTasks,
        current_df=base_df,
        task_overrides={"top_n": 1},
    )
    ctx.metadata["task_durations"] = durations
    result = ctx.run_task(task)

    top = result.summary["top_bottlenecks"]
    assert len(top) == 1
    assert isinstance(top[0]["duration_sec"], float)
    assert round(top[0]["duration_sec"], 4) == 1.2346
