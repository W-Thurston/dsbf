# tests/test_utils/test_report_utils.py

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.utils.report_utils import render_user_report, write_metadata_report


@pytest.fixture
def sample_context(tmp_path):
    df = pd.DataFrame()
    context = AnalysisContext(data=df, output_dir=str(tmp_path))

    # Inject dummy metadata
    context.run_metadata = {
        "engine": "ProfileEngine",
        "timestamp": "test_run",
        "dsbf_version": "0.1.0",
        "python": "3.11.0",
    }
    context.metadata = {
        "task_durations": {"DetectSkewness": 0.12},
        "run_stats": {"total_tasks": 2},
    }

    # Inject both EDA and diagnostic results
    context.results = {
        "DetectSkewness": TaskResult(
            name="DetectSkewness", status="success", summary={"mean": 0.1}
        ),
        "IdentifyBottleneckTasks": TaskResult(
            name="IdentifyBottleneckTasks",
            status="success",
            summary={"top_bottlenecks": []},
        ),
        "LogResourceUsage": TaskResult(
            name="LogResourceUsage",
            status="success",
            summary={"total_runtime_sec": 5.0},
        ),
    }

    return context


def test_render_user_report_filters_diagnostics(sample_context):
    report_path = os.path.join(sample_context.output_dir, "report.json")
    render_user_report(sample_context.results, output_path=report_path)

    with open(report_path, "r") as f:
        report = json.load(f)

    assert "results" in report
    assert "DetectSkewness" in report["results"]
    assert "IdentifyBottleneckTasks" not in report["results"]
    assert "LogResourceUsage" not in report["results"]


def test_write_metadata_report_includes_diagnostics(sample_context):
    meta_path = os.path.join(sample_context.output_dir, "metadata_report.json")
    write_metadata_report(sample_context, filename="metadata_report.json")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    assert "run_stats" in metadata
    assert "task_durations" in metadata
    assert "diagnostic_results" in metadata

    diagnostics = metadata["diagnostic_results"]
    assert "IdentifyBottleneckTasks" in diagnostics
    assert "LogResourceUsage" in diagnostics

    assert diagnostics["IdentifyBottleneckTasks"]["status"] == "success"


def test_write_metadata_handles_missing_diagnostics(tmp_path):
    df = pd.DataFrame()
    context = AnalysisContext(data=df, output_dir=str(tmp_path))
    context.run_metadata = {"engine": "ProfileEngine"}
    context.metadata = {"run_stats": {}}
    context.results = {
        "DetectSkewness": TaskResult(
            name="DetectSkewness", status="success", summary={"info": "ok"}
        )
    }

    path = os.path.join(str(tmp_path), "metadata_report.json")
    write_metadata_report(context, filename="metadata_report.json")

    with open(path, "r") as f:
        metadata = json.load(f)

    assert "diagnostic_results" in metadata
    assert metadata["diagnostic_results"] == {}  # should be empty, not missing


def test_write_metadata_report_includes_memory_and_peak(tmp_path: Path):
    df = pd.DataFrame()
    context = AnalysisContext(data=df, output_dir=str(tmp_path))
    context.run_metadata = {
        "engine": "ProfileEngine",
        "timestamp": "memory_test",
    }
    context.metadata = {
        "task_memory": {
            "TaskA": 42.0,
            "TaskB": 87.5,
        },
        "peak_memory_mb": 87.5,
        "task_durations": {"TaskA": 0.1, "TaskB": 0.2},
        "run_stats": {"total_tasks": 2},
    }
    context.results = {
        "TaskA": TaskResult(name="TaskA", status="success", summary={}),
        "TaskB": TaskResult(name="TaskB", status="success", summary={}),
    }
    out_path = tmp_path / "metadata_report.json"
    write_metadata_report(context, filename=out_path.name)

    with out_path.open() as f:
        report = json.load(f)

    assert "task_memory" in report
    assert "peak_memory_mb" in report
    assert report["task_memory"]["TaskB"] == 87.5
    assert report["peak_memory_mb"] == 87.5
