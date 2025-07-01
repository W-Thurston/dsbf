# tests/eda/test_log_reliability_warnings.py

from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    log_reliability_warnings,
)


class DummyTask:
    name = "dummy_task"

    def __init__(self):
        self.logs = []

    def _log(self, msg, level="info"):
        self.logs.append((level, msg))
        print(f"[{level.upper()}] {msg}")


def test_log_reliability_warnings_outputs_expected_messages(capfd):
    result = TaskResult(name="dummy_task")

    add_reliability_warning(
        result,
        level="heuristic_caution",
        code="test_flag",
        description="Something might be off.",
        recommendation="Consider double-checking this.",
    )

    task = DummyTask()
    log_reliability_warnings(task, result)
    out, _ = capfd.readouterr()

    assert "HEURISTIC_CAUTION" in out
    assert "test_flag" in out
    assert "Something might be off." in out
    assert "Consider double-checking" in out
