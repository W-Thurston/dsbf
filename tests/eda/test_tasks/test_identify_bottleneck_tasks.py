# tests/eda/test_tasks/test_identify_bottleneck_tasks.py

from typing import TYPE_CHECKING

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.tasks.identify_bottleneck_tasks import IdentifyBottleneckTasks
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def _ctx_with_durations(df, tmp_path, durations: dict) -> AnalysisContext:
    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    ctx.set_metadata("task_durations", durations)
    return ctx


def test_top_n_slowest_tasks_returned(tmp_path) -> None:
    """The top-N slowest tasks must be returned in descending duration order."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = _ctx_with_durations(
        df,
        tmp_path,
        {"fast": 0.1, "medium": 2.0, "slow": 8.0, "very_slow": 12.0, "moderate": 1.5},
    )

    task = IdentifyBottleneckTasks()
    task.set_input(df)
    task.context = ctx
    ctx.set_metadata(
        "task_durations",
        {"fast": 0.1, "medium": 2.0, "slow": 8.0, "very_slow": 12.0, "moderate": 1.5},
    )

    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    bottlenecks = result.summary["top_bottlenecks"]
    assert len(bottlenecks) <= 5
    assert bottlenecks[0]["task"] == "very_slow"
    assert bottlenecks[1]["task"] == "slow"
    # Verify descending order
    durations: list = [b["duration_sec"] for b in bottlenecks]
    assert durations == sorted(durations, reverse=True)


def test_top_n_configurable(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    durations: dict[str, float] = {f"task_{i}": float(i) for i in range(10)}
    ctx: AnalysisContext = _ctx_with_durations(df, tmp_path, durations)

    ctx, task = make_ctx_and_task(
        task_cls=IdentifyBottleneckTasks,
        current_df=df,
        task_overrides={"top_n": 3},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("task_durations", durations)
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert len(result.summary["top_bottlenecks"]) == 3


def test_failed_when_no_durations(tmp_path) -> None:
    """Task must return 'failed' status when no duration metadata is available."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    # No task_durations set

    task = IdentifyBottleneckTasks()
    task.set_input(df)
    task.context = ctx

    result: TaskResult = ctx.run_task(task)

    assert result.status == "failed"
    assert "no task durations" in result.summary["message"].lower()


def test_recommendation_for_slow_tasks(tmp_path) -> None:
    """Recommendations must be emitted for tasks exceeding 5 seconds."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = _ctx_with_durations(df, tmp_path, {"heavy_task": 10.0, "light_task": 0.1})

    task = IdentifyBottleneckTasks()
    task.set_input(df)
    task.context = ctx

    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.recommendations is not None
    assert any("heavy_task" in r for r in result.recommendations)


def test_no_recommendation_for_fast_tasks(tmp_path) -> None:
    """No recommendations when all tasks complete under 5 seconds."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = _ctx_with_durations(df, tmp_path, {"a": 0.5, "b": 1.2, "c": 0.8})

    task = IdentifyBottleneckTasks()
    task.set_input(df)
    task.context = ctx

    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.recommendations == [] or result.recommendations is None


def test_no_plots_generated(tmp_path) -> None:
    """Bottleneck identification must not generate plots."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = _ctx_with_durations(df, tmp_path, {"task_a": 1.0})

    task = IdentifyBottleneckTasks()
    task.set_input(df)
    task.context = ctx

    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
