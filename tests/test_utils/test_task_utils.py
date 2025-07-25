# tests/test_utils/test_task_utils.py
import json
import os
import tempfile

import pytest

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import TASK_REGISTRY, register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.task_utils import (
    filter_tasks,
    is_diagnostic_task,
    validate_task_result,
    write_task_metadata,
)


def test_is_diagnostic_task():
    assert is_diagnostic_task("IdentifyBottleneckTasks") is True
    assert is_diagnostic_task("LogResourceUsage") is True
    assert is_diagnostic_task("DetectSkewness") is False
    assert is_diagnostic_task("SummarizeTextFields") is False
    assert is_diagnostic_task("unknown_task") is False


@pytest.mark.parametrize(
    "criteria,expected_subset",
    [
        (
            {"domain": "core"},
            {name for name, spec in TASK_REGISTRY.items() if spec.domain == "core"},
        ),
        (
            {"runtime_estimate": ["fast", "moderate"]},
            {
                name
                for name, spec in TASK_REGISTRY.items()
                if spec.runtime_estimate in {"fast", "moderate"}
            },
        ),
        (
            {"tags": ["text"]},
            {
                name
                for name, spec in TASK_REGISTRY.items()
                if spec.tags and "text" in spec.tags
            },
        ),
        (
            {"stage": "cleaned"},
            {name for name, spec in TASK_REGISTRY.items() if spec.stage == "cleaned"},
        ),
        (
            {"tags": ["nonexistent"]},
            set(),
        ),
    ],
)
def test_filter_tasks_matches_expected(criteria, expected_subset):
    result = set(filter_tasks(criteria))
    assert expected_subset.issubset(result)


def test_write_task_metadata_creates_valid_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "task_metadata.json")
        write_task_metadata(output_path)

        assert os.path.exists(output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert len(data) > 0

        first_task = next(iter(data.values()))
        assert "domain" in first_task
        assert "runtime_estimate" in first_task
        assert "summary" in first_task


@register_task(name="test_task", description="TEST", domain="test", tags=["foo"])
class DummyTask(BaseTask):
    def run(self):
        self.output = TaskResult(name=self.name, summary={"message": "ok"})


def test_validate_task_result_passes():
    result = TaskResult(name="test_task", status="success")
    assert validate_task_result(result)


def test_validate_task_result_fails_on_score():
    result = TaskResult(name="test_task", ml_impact_score=1.5)
    assert not validate_task_result(result)
