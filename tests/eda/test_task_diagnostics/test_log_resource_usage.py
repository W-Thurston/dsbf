# tests/eda/test_task_diagnostics/test_log_resource_usage.py

from pathlib import Path

import pandas as pd
import pytest

from dsbf.eda.tasks.log_resource_usage import LogResourceUsage
from tests.helpers.context_utils import make_ctx_and_task


@pytest.fixture
def base_df():
    return pd.DataFrame()


@pytest.fixture
def sample_durations():
    return {
        "load_data": 0.8,
        "detect_outliers": 1.6,
        "summarize_text_fields": 2.4,
        "generate_report": 3.2,
    }


def test_usage_task_outputs_total_and_mean(base_df, sample_durations, tmp_path):
    ctx, task = make_ctx_and_task(
        task_cls=LogResourceUsage,
        current_df=base_df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.metadata["task_durations"] = sample_durations

    result = ctx.run_task(task)

    assert result.status == "success"
    summary = result.summary
    assert summary["task_count"] == len(sample_durations)
    assert round(summary["total_runtime_sec"], 2) == round(
        sum(sample_durations.values()), 2
    )
    assert summary["mean_task_time"] == round(
        summary["total_runtime_sec"] / summary["task_count"], 4
    )
    assert isinstance(summary["task_durations"], dict)
    assert all(isinstance(v, float) for v in summary["task_durations"].values())


def test_zero_tasks_defaults_to_empty_output(base_df, tmp_path):
    ctx, task = make_ctx_and_task(
        task_cls=LogResourceUsage,
        current_df=base_df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.metadata["task_durations"] = {}

    result = ctx.run_task(task)

    summary = result.summary
    assert result.status == "success"
    assert summary["task_count"] == 0
    assert summary["mean_task_time"] is None
    assert summary["total_runtime_sec"] == 0.0


def test_runtime_over_30sec_triggers_recommendation(base_df, tmp_path):
    durations = {f"task_{i}": 6.5 for i in range(6)}  # Total = 39.0

    ctx, task = make_ctx_and_task(
        task_cls=LogResourceUsage,
        current_df=base_df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.metadata["task_durations"] = durations
    result = ctx.run_task(task)

    recs = result.recommendations
    assert recs is not None
    assert any("exceeds 30 seconds" in r for r in recs)


def test_high_mean_runtime_triggers_recommendation(base_df, tmp_path):
    durations = {
        "slow_task_1": 8.0,
        "slow_task_2": 7.5,
    }

    ctx, task = make_ctx_and_task(
        task_cls=LogResourceUsage,
        current_df=base_df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.metadata["task_durations"] = durations
    result = ctx.run_task(task)

    recs = result.recommendations
    assert recs is not None
    assert any("average runtime" in r.lower() for r in recs)


def test_resource_usage_plot_generated(base_df, sample_durations, tmp_path):
    ctx, task = make_ctx_and_task(
        task_cls=LogResourceUsage,
        current_df=base_df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.metadata["task_durations"] = sample_durations

    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is not None
    assert "task_runtime" in result.plots

    plot_entry = result.plots["task_runtime"]
    static_path = plot_entry["static"]
    interactive = plot_entry["interactive"]

    assert isinstance(static_path, Path)
    assert static_path.exists()
    assert static_path.suffix == ".png"

    assert interactive["type"] == "bar"
    assert "annotations" in interactive
    assert all(":" in a and a.endswith("s") for a in interactive["annotations"])
