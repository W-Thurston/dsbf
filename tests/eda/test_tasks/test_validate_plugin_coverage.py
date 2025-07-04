# tests/eda/test_tasks/test_validate_plugin_coverage.py

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.validate_plugin_coverage import ValidatePluginCoverageTask
from tests.helpers.context_utils import make_ctx_and_task


def test_plugin_coverage_no_warnings():
    ctx, task = make_ctx_and_task(
        task_cls=ValidatePluginCoverageTask,
        current_df=[],
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.reliability_warnings is None
    assert "registered" in result.summary["message"]


def test_plugin_coverage_with_warnings():
    ctx, task = make_ctx_and_task(task_cls=ValidatePluginCoverageTask, current_df=[])
    ctx.set_metadata(
        "plugin_warnings",
        [
            {"file": "plugins/a.py", "message": "No tasks registered"},
            {"file": "plugins/b.py", "message": "No tasks registered"},
        ],
    )

    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.reliability_warnings is not None
    assert "plugin_registration" in result.reliability_warnings
    assert "missing_tasks" in result.reliability_warnings["plugin_registration"]
    assert "did not register" in result.summary["message"]
