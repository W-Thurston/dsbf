# tests/eda/test_task_result.py

from pathlib import Path

from dsbf.eda.task_result import TaskResult


def test_task_result_to_dict():
    result = TaskResult(
        name="test_task",
        status="success",
        summary={"message": "All good"},
        data={"key": "value"},
        plots=[Path("/tmp/plot1.png"), Path("/tmp/plot2.png")],
        metadata={"runtime": 1.23},
    )

    result_dict = result.to_dict()

    assert result_dict["name"] == "test_task"
    assert result_dict["status"] == "success"
    assert result_dict["summary"] == {"message": "All good"}
    assert result_dict["data"] == {"key": "value"}
    assert result_dict["plots"] == ["/tmp/plot1.png", "/tmp/plot2.png"]
    assert result_dict["metadata"]["runtime"] == 1.23


def test_task_result_str():
    result = TaskResult(
        name="null_check", status="skipped", summary={"message": "No nulls found"}
    )
    assert str(result) == (
        "TaskResult(name=null_check, status=skipped,"
        " summary={'message': 'No nulls found'})"
    )


def test_task_result_defaults_and_empty_fields():
    result = TaskResult(
        name="no_data", status="success", summary={"message": "No nulls found"}
    )
    result_dict = result.to_dict()

    assert result_dict["data"] is None
    assert result_dict["plots"] is None
    assert isinstance(result_dict["metadata"], dict)


def test_task_result_failure_metadata_fields():
    result = TaskResult(
        name="fail_task",
        status="failed",
        summary={"message": "Failure during task execution"},
        error_metadata={
            "error_type": "KeyError",
            "trace_summary": "Column 'foo' not found",
            "suggested_action": "Check for renamed or dropped columns",
        },
    )

    assert result.status == "failed"
    assert result.error_metadata is not None
    assert result.error_metadata["error_type"] == "KeyError"
    assert "foo" in result.error_metadata["trace_summary"]
    assert "Check" in result.error_metadata["suggested_action"]
