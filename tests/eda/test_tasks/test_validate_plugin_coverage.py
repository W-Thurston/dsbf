# tests/eda/test_tasks/test_validate_plugin_coverage.py

from typing import TYPE_CHECKING

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.tasks.validate_plugin_coverage import ValidatePluginCoverageTask

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_passes_when_no_warnings(tmp_path) -> None:
    """Task must return success with a clean message when no plugin warnings exist."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    # No plugin_warnings set in metadata

    task = ValidatePluginCoverageTask()
    task.set_input(df)
    task.context = ctx
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["plugin_warnings"] == []
    assert "successfully" in result.summary["message"].lower()


def test_reports_plugin_warnings(tmp_path) -> None:
    """Task must report plugin files that registered no tasks."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    ctx.set_metadata("plugin_warnings", ["plugins/empty_plugin.py", "plugins/bad.py"])

    task = ValidatePluginCoverageTask()
    task.set_input(df)
    task.context = ctx
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert len(result.data["plugin_warnings"]) == 2
    assert "2 plugin file" in result.summary["message"]


def test_reliability_warning_emitted_when_warnings_present(tmp_path) -> None:
    """A reliability warning must be attached when plugin warnings exist."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    ctx.set_metadata("plugin_warnings", ["plugins/missing.py"])

    task = ValidatePluginCoverageTask()
    task.set_input(df)
    task.context = ctx
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.reliability_warnings is not None
    assert len(result.reliability_warnings) > 0
    # reliability_warnings is a list of strings - check content rather than keys
    assert any(
        "missing" in str(w).lower() or "plugin" in str(w).lower()
        for w in result.reliability_warnings
    )


def test_no_reliability_warning_when_clean(tmp_path) -> None:
    """No reliability warning must be emitted when all plugins registered tasks."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))

    task = ValidatePluginCoverageTask()
    task.set_input(df)
    task.context = ctx
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert not result.reliability_warnings
