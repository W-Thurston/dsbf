# tests/eda/test_execution_graph_failure.py

from dsbf.core.base_task import BaseTask
from dsbf.core.context import AnalysisContext
from dsbf.eda.graph import ExecutionGraph, Task


class FailingTask(BaseTask):
    def run(self):
        raise ValueError("Simulated failure in FailingTask")


def test_execution_graph_handles_failure():
    context = AnalysisContext(data={})
    task = FailingTask(name="fail_test")
    graph = ExecutionGraph([Task("fail_test", task)])

    results = graph.run(context)

    result = context.get_result("fail_test")
    assert results is not None
    assert result is not None
    assert result.status == "failed"
    assert result.error_metadata is not None
    assert result.error_metadata["error_type"] == "RuntimeError"
    assert "Simulated failure" in result.error_metadata["trace_summary"]
