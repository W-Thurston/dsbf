"""The OpenAPI document is exported, current, and guarded against breaking changes."""

import copy
import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dsbf.interfaces import openapi_export
from dsbf.interfaces.cli import app
from dsbf.model.schema import Change, run_schema, schema_text
from dsbf.service import API_VERSION, openapi_document

pytestmark = pytest.mark.unit

runner = CliRunner()
GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com"]
RUN_PATH = "/v2/runs/{run_id}/findings"


def edited(edit) -> dict:
    document = copy.deepcopy(openapi_document())
    edit(document)
    return document


def changes(edit) -> list[Change]:
    return openapi_export.compare_openapi(openapi_document(), edited(edit))


def get_findings(document):
    return document["paths"][RUN_PATH]["get"]


def test_an_unchanged_document_has_no_changes():
    assert changes(lambda d: None) == []


def test_every_read_endpoint_has_a_named_operation():
    operations = {
        operation["operationId"]
        for item in openapi_document()["paths"].values()
        for operation in item.values()
    }
    assert operations >= {
        "listDatasets",
        "listDatasetRuns",
        "getFindingHistory",
        "compareRuns",
        "getRun",
        "listFindings",
        "getColumnProfile",
        "findColumnProfile",
        "getCheckResult",
        "getSnapshot",
        "getArtifact",
    }


def test_a_removed_path_or_operation_is_breaking():
    gone = changes(lambda d: d["paths"].pop(RUN_PATH))
    assert gone == [Change(RUN_PATH, True, "path removed")]
    no_get = changes(lambda d: d["paths"][RUN_PATH].pop("get"))
    assert no_get == [Change(f"GET {RUN_PATH}", True, "operation removed")]


def test_parameter_changes_are_classified():
    def drop_limit(d):
        get_findings(d)["parameters"] = [
            p for p in get_findings(d)["parameters"] if p["name"] != "limit"
        ]

    def require_code(d):
        for p in get_findings(d)["parameters"]:
            if p["name"] == "code":
                p["required"] = True

    def add_optional(d):
        get_findings(d)["parameters"].append(
            {"name": "extra", "in": "query", "required": False, "schema": {}}
        )

    def add_required(d):
        get_findings(d)["parameters"].append(
            {"name": "extra", "in": "query", "required": True, "schema": {}}
        )

    assert [c.breaking for c in changes(drop_limit)] == [True]
    assert [c.breaking for c in changes(require_code)] == [True]
    assert [c.breaking for c in changes(add_optional)] == [False]
    assert [c.breaking for c in changes(add_required)] == [True]


def test_a_response_field_removed_is_breaking_and_one_added_is_not():
    def remove_total(d):
        del d["components"]["schemas"]["RunOverview"]["properties"]["n_findings"]

    def add_field(d):
        d["components"]["schemas"]["RunOverview"]["properties"]["remark"] = {
            "type": "string"
        }

    removed = changes(remove_total)
    assert [(c.breaking, c.message) for c in removed] == [(True, "property removed")]
    added = changes(add_field)
    assert [c.breaking for c in added] == [False]


def test_a_removed_response_status_is_breaking():
    gone = changes(lambda d: get_findings(d)["responses"].pop("404"))
    assert gone == [Change(f"GET {RUN_PATH} 404", True, "response removed")]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "schema" / openapi_export.OPENAPI_FILE
    target.parent.mkdir()
    target.write_text(openapi_export.openapi_text(), encoding="utf-8")
    run_file = target.with_name("run.schema.json")
    run_file.write_text(schema_text(run_schema()), encoding="utf-8")
    for command in (["init", "-q", "-b", "main"], ["add", "-A"]):
        subprocess.run([*GIT, *command], cwd=tmp_path, check=True)
    subprocess.run([*GIT, "commit", "-q", "-m", "publish"], cwd=tmp_path, check=True)
    return target


def check():
    schema = Path("schema") / "run.schema.json"  # git reads it relative to the repo
    return runner.invoke(
        app, ["schema", "check", "--base", "main", "--schema", str(schema)]
    )


def publish(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", "utf-8")


def drop_path(document: dict) -> None:
    del document["paths"][RUN_PATH]


def test_the_gate_fails_an_unannounced_removal_and_passes_a_bumped_one(repo):
    gone = edited(drop_path)
    publish(repo, gone)
    result = check()
    assert result.exit_code == 1, result.output
    assert (
        "openapi: breaking: /v2/runs/{run_id}/findings: path removed" in result.output
    )
    assert "bump API_VERSION" in result.output

    major = int(API_VERSION.split(".")[0]) + 1
    gone["info"]["version"] = f"{major}.0.0"
    publish(repo, gone)
    assert check().exit_code == 0


def test_the_committed_document_is_current():
    out = Path(__file__).parents[2] / "schema"
    result = runner.invoke(app, ["schema", "export", "--check", "--out", str(out)])
    assert result.exit_code == 0, result.output
