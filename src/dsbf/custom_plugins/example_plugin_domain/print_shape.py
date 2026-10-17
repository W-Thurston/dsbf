# dsbf/custom_plugins/example_plugin_domain/print_shape.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult


@register_task(
    name="print_shape_debug",
    tags=["debug"],
    domain="internal",
    stage="raw",
    runtime_estimate="fast",
    profiling_depth="basic",
    description="Prints the shape of the input data for debugging.",
)
class PrintShapeDebug(BaseTask):
    def run(self):
        df = self.input_data
        shape = df.shape if hasattr(df, "shape") else (None, None)
        self._log(f"[print_shape_debug] Data shape: {shape}", level="info")
        self.output = TaskResult(
            name=self.name, summary={"message": f"Data has shape {shape}"}
        )
