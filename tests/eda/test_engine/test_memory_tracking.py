# tests/eda/test_engine/test_memory_tracking.py

import polars as pl

from dsbf.eda.graph import ExecutionGraph, Task
from dsbf.eda.task_result import TaskResult
from tests.helpers.context_utils import make_ctx_and_task


class MemoryTestTask:
    def __init__(self, name="memory_test", config=None):
        self.name = name
        self.config = config or {}
        self.context = None
        self.output = None

    def set_input(self, input_data):
        self.input_data = input_data

    def get_output(self):
        return self.output

    def _log(self, msg, level="info"):
        pass

    def run(self):
        big_list = [0] * 5_000_000
        self.output = TaskResult(name=self.name, summary={"message": "done"})
        self.output.data = {"payload": big_list}


def test_memory_limit_exceeded_warning():
    df = pl.DataFrame({"x": range(100)})
    overrides = {"resource_limits": {"max_memory_gb": 0.01}}

    ctx, dummy_task = make_ctx_and_task(
        task_cls=MemoryTestTask,
        current_df=df,
        global_overrides=overrides,
    )

    task = Task(name=dummy_task.name, task_instance=dummy_task)
    graph = ExecutionGraph([task])
    results = graph.run(ctx)

    mem_used = ctx.metadata["task_memory"][dummy_task.name]
    assert mem_used > 10.0
    assert results[dummy_task.name].metadata.get("memory_exceeded") is True
    assert "peak_memory_mb" in ctx.metadata
