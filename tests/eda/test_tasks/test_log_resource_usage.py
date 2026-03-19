# tests/eda/test_tasks/test_log_resource_usage.py

from typing import TYPE_CHECKING

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.tasks.log_resource_usage import LogResourceUsage

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def _ctx_with_durations(df, tmp_path, durations: dict) -> AnalysisContext:
    """Return a context with pre-populated task_durations metadata."""
    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    ctx.set_metadata("task_durations", durations)
    return ctx


def test_totals_computed_correctly(tmp_path) -> None:
    """total_runtime_sec must equal the sum of all task durations."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = _ctx_with_durations(df, tmp_path, {"task_a": 1.5, "task_b": 2.0})
    task = LogResourceUsage()
    task.set_input(df)
    task.context = ctx

    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["total_runtime_sec"] == 3.5
    assert result.summary["task_count"] == 2
    assert result.summary["mean_task_time"] == 1.75


def test_mean_task_time_none_when_no_tasks(tmp_path) -> None:
    """mean_task_time must be None when there are no recorded durations."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = _ctx_with_durations(df, tmp_path, {})
    # Inject run_stats fallback
    ctx.set_metadata("run_stats", {"duration": 5.0})
    task = LogResourceUsage()
    task.set_input(df)
    task.context = ctx

    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["task_count"] == 0
    assert result.summary["mean_task_time"] is None
    assert result.summary["total_runtime_sec"] == 5.0


def test_recommendation_emitted_for_slow_run(tmp_path) -> None:
    """A recommendation must be emitted when total runtime exceeds 30 seconds."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = _ctx_with_durations(df, tmp_path, {"task_a": 20.0, "task_b": 15.0})
    task = LogResourceUsage()
    task.set_input(df)
    task.context = ctx

    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.recommendations is not None
    assert any("30" in r or "caching" in r.lower() for r in result.recommendations)


def test_no_recommendation_for_fast_run(tmp_path) -> None:
    """No recommendations must be emitted when the run is within acceptable limits."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = _ctx_with_durations(df, tmp_path, {"task_a": 0.5, "task_b": 0.3})
    task = LogResourceUsage()
    task.set_input(df)
    task.context = ctx

    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.recommendations == [] or result.recommendations is None


def test_task_durations_sorted_ascending(tmp_path) -> None:
    """task_durations in the summary must be sorted by ascending duration."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = _ctx_with_durations(df, tmp_path, {"slow": 5.0, "fast": 0.1, "medium": 2.5})
    task = LogResourceUsage()
    task.set_input(df)
    task.context = ctx

    result: TaskResult = ctx.run_task(task)

    durations: list = list(result.summary["task_durations"].values())
    assert durations == sorted(durations)


def test_no_plots_generated(tmp_path) -> None:
    """Log resource usage must not generate plots."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = _ctx_with_durations(df, tmp_path, {"task_a": 1.0})
    task = LogResourceUsage()
    task.set_input(df)
    task.context = ctx

    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
