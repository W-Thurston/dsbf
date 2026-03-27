# dsbf/eda/tasks/validate_plugin_coverage.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    log_reliability_warnings,
)


@register_task(
    name="validate_plugin_coverage",
    display_name="Plugin Coverage Check",
    description="Checks that all loaded plugin files registered at least one task.",
    profiling_depth="basic",
    stage="any",
    phase="diagnostic",
    domain="core",
    tags=["plugin", "diagnostic", "core"],
    runtime_estimate="fast",
    expected_semantic_types=["any"],
)
class ValidatePluginCoverageTask(BaseTask):
    """
    Validate that every loaded plugin file registered at least one task.

    Reads ``plugin_warnings`` from the analysis context metadata - a list of
    plugin file paths that were loaded but did not register any tasks via
    ``@register_task``. Emits a reliability warning if any are found.

    This task is diagnostic infrastructure - it runs after all plugin files
    are loaded and surfaces silent registration failures that would otherwise
    cause tasks to be missing from the DAG with no error.
    """

    def run(self) -> None:
        """
        Check plugin registration and populate self.output.

        Raises:
            RuntimeError: If no analysis context is attached.

        """
        if self.context is None:
            raise RuntimeError("Task context not set before run()")

        warnings: list = self.context.get_metadata("plugin_warnings", [])
        msg: str = (
            f"{len(warnings)} plugin file(s) did not register any tasks."
            if warnings
            else "All plugin files registered tasks successfully."
        )

        self.output = TaskResult(
            name=self.name,
            status="success",
            summary={"message": msg},
            data={"plugin_warnings": warnings},
        )

        if warnings:
            add_reliability_warning(
                result=self.output,
                level="plugin_registration",
                code="missing_tasks",
                description=msg,
                recommendation=(
                    "Check your plugin files for missing @register_task "
                    "decorators or import issues."
                ),
            )

        log_reliability_warnings(self, self.output)
