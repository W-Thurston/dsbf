# tests/utils/test_task_utils.py

from dsbf.utils.task_utils import is_diagnostic_task


def test_is_diagnostic_task():
    assert is_diagnostic_task("IdentifyBottleneckTasks") is True
    assert is_diagnostic_task("LogResourceUsage") is True
    assert is_diagnostic_task("DetectSkewness") is False
    assert is_diagnostic_task("SummarizeTextFields") is False
    assert is_diagnostic_task("unknown_task") is False
