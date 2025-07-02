# tests/test_task_result_validation.py

from dsbf.eda.task_result import TaskResult
from dsbf.utils.task_result_validator import validate_task_result


def test_valid_result_passes():
    result = TaskResult(
        name="test_task", status="success", recommendations=["Check skewness"]
    )
    assert validate_task_result(result)


def test_invalid_status_fails():
    result = TaskResult(name="bad_status")
    object.__setattr__(result, "status", "done")  # invalid
    assert not validate_task_result(result)


def test_invalid_recommendations_type():
    result = TaskResult(name="bad_recs", status="success")
    object.__setattr__(result, "recommendations", "Do better")  # not a list
    assert not validate_task_result(result)


def test_invalid_recommendations_list_elements():
    result = TaskResult(name="mixed_recs", status="success")
    object.__setattr__(result, "recommendations", ["good", 123])  # mixed types
    assert not validate_task_result(result)


def test_validates_ml_impact_score_bounds(monkeypatch):
    result = TaskResult(name="scored", status="success")

    # Dynamically attach a valid score
    setattr(result, "ml_impact_score", 0.85)
    assert validate_task_result(result)

    setattr(result, "ml_impact_score", 1.2)
    assert not validate_task_result(result)

    setattr(result, "ml_impact_score", -0.1)
    assert not validate_task_result(result)
