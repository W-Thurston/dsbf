"""The example GitHub Actions workflow is valid and its commands do what it claims."""

import os
import shlex
from pathlib import Path

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from dsbf.interfaces.cli import app
from tests.test_docs.conftest import write_customers

ROOT = Path(__file__).parents[2]
WORKFLOW = ROOT / "examples" / "github-actions" / "dsbf-check.yml"

pytestmark = pytest.mark.integration

FAILING = """\
version: 1
rules:
  - name: heavy-tails
    when: { code: heavy_tails }
    then: fail
"""
PASSING = """\
version: 1
rules:
  - name: heavy-tails
    when: { code: heavy_tails }
    then: warn
"""


def steps() -> list[dict]:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return document["jobs"]["dsbf"]["steps"]


def check_command(workflow_env: dict[str, str]) -> list[str]:
    step = next(s for s in steps() if s.get("id") == "check")
    expanded = os.path.expandvars(step["run"])
    for name, value in workflow_env.items():
        expanded = expanded.replace(f"${name}", value)
    return shlex.split(expanded)


def prepare(tmp_path, monkeypatch, policy: str) -> list[str]:
    env = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["env"]
    (tmp_path / "data").mkdir()
    write_customers(tmp_path / env["DATA"])
    (tmp_path / env["POLICY"]).write_text(policy, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DSBF_HOME", str(tmp_path / "home"))
    return check_command(env)


def test_the_workflow_keeps_the_findings_even_when_the_check_fails():
    by_name = {s["name"]: s for s in steps() if "name" in s}

    assert by_name["Check the data against the policy"]["continue-on-error"] is True
    assert "failure" in by_name["Fail the job if the policy failed"]["if"]
    assert "dsbf-results" in by_name["Keep the findings"]["with"]["path"]


@pytest.mark.parametrize(
    ("policy", "exit_code"), [(FAILING, 1), (PASSING, 0)], ids=["fails", "passes"]
)
def test_the_check_step_writes_findings_and_exits_by_policy(
    tmp_path, monkeypatch, policy, exit_code
):
    command = prepare(tmp_path, monkeypatch, policy)
    assert command[0] == "dsbf"

    result = CliRunner().invoke(app, command[1:])

    assert result.exit_code == exit_code, result.output
    findings = pl.read_parquet(tmp_path / "dsbf-results" / "findings.parquet")
    assert findings.height > 0
    assert (tmp_path / "dsbf-results" / "run.json").is_file()
