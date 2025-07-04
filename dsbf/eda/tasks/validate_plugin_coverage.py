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
    domain="core",
    tags=["plugin", "diagnostic", "core"],
    runtime_estimate="fast",
    inputs=["context"],
    outputs=["TaskResult"],
)
class ValidatePluginCoverageTask(BaseTask):
    def run(self):
        if self.context is None:
            raise RuntimeError("Task context not set before run()")

        warnings = self.context.get_metadata("plugin_warnings", [])
        msg = (
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
                    "Check your plugin files for missing @register_task"
                    " decorators or import issues."
                ),
            )

        log_reliability_warnings(self, self.output)
