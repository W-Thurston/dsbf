# tests/eda/test_task_result.py

from pathlib import Path

from dsbf.eda.task_result import TaskResult, add_reliability_warning


def test_task_result_to_dict():
    result = TaskResult(
        name="test_task",
        status="success",
        summary={"message": "All good"},
        data={"key": "value"},
        plots={
            "plot1": {
                "static": Path("/tmp/plot1.png"),
                "interactive": {
                    "type": "bar",
                    "config": {},
                    "data": {},
                    "annotations": [],
                },
            },
            "plot2": {
                "static": Path("/tmp/plot2.png"),
                "interactive": {
                    "type": "line",
                    "config": {},
                    "data": {},
                    "annotations": [],
                },
            },
        },
        metadata={"runtime": 1.23},
    )

    result_dict = result.to_dict()

    assert result_dict["name"] == "test_task"
    assert result_dict["status"] == "success"
    assert result_dict["summary"] == {"message": "All good"}
    assert result_dict["data"] == {"key": "value"}

    assert isinstance(result_dict["plots"], dict)
    assert result_dict["plots"]["plot1"]["static"] == "/tmp/plot1.png"
    assert result_dict["plots"]["plot2"]["static"] == "/tmp/plot2.png"

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


def test_add_reliability_warning_creates_nested_dict_structure():
    result = TaskResult(name="test")

    add_reliability_warning(
        result,
        level="strong_warning",
        code="low_row_count",
        description="Sample size < 30",
        recommendation="Use bootstrap CIs",
    )

    assert result.reliability_warnings is not None
    assert "strong_warning" in result.reliability_warnings
    assert "low_row_count" in result.reliability_warnings["strong_warning"]

    entry = result.reliability_warnings["strong_warning"]["low_row_count"]
    assert entry["description"] == "Sample size < 30"
    assert entry["recommendation"] == "Use bootstrap CIs"


def test_add_multiple_warning_levels_and_codes():
    result = TaskResult(name="test")

    add_reliability_warning(
        result,
        level="strong_warning",
        code="zero_variance",
        description="Stdev ~ 0",
        recommendation="Drop constant features",
    )
    add_reliability_warning(
        result,
        level="heuristic_caution",
        code="high_skew",
        description="Highly skewed columns",
        recommendation="Try Spearman rank correlation",
    )

    assert result.reliability_warnings is not None
    assert set(result.reliability_warnings.keys()) == {
        "strong_warning",
        "heuristic_caution",
    }
    assert "zero_variance" in result.reliability_warnings["strong_warning"]
    assert "high_skew" in result.reliability_warnings["heuristic_caution"]


def test_warning_serializes_in_to_dict():
    result = TaskResult(name="test")

    add_reliability_warning(
        result,
        level="strong_warning",
        code="tiny_sample",
        description="N too small",
        recommendation="Use with caution",
    )

    result_dict = result.to_dict()
    assert "reliability_warnings" in result_dict
    assert (
        result_dict["reliability_warnings"]["strong_warning"]["tiny_sample"][
            "description"
        ]
        == "N too small"
    )
