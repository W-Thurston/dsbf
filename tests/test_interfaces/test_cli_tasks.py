# tests/test_interfaces/test_cli_tasks.py
"""``dsbf tasks export`` writes the task metadata only when asked to."""

import json

import pytest
from typer.testing import CliRunner

from dsbf.interfaces.cli import app

pytestmark = pytest.mark.unit

runner = CliRunner()


def test_export_prints_every_registered_task_as_json():
    result = runner.invoke(app, ["tasks", "export"])

    assert result.exit_code == 0
    metadata = json.loads(result.stdout)
    assert "summarize_dataset_shape" in metadata
    assert set(metadata["summarize_dataset_shape"]) >= {"domain", "tags", "stage"}


def test_export_writes_the_file_given_by_output(tmp_path):
    target = tmp_path / "nested" / "task_metadata.json"

    result = runner.invoke(app, ["tasks", "export", "--output", str(target)])

    assert result.exit_code == 0
    assert "summarize_dataset_shape" in json.loads(target.read_text())
