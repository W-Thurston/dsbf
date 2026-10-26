"""``dsbf schema`` writes the published files and guards them against breaking."""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dsbf.interfaces import schema_export
from dsbf.interfaces.cli import app
from dsbf.model import SCHEMA_VERSION
from dsbf.model.schema import policy_schema, run_schema, schema_text

runner = CliRunner()

# The version a schema change is announced with, whatever the current one is.
_MAJOR, _MINOR, _ = SCHEMA_VERSION.split(".")
NEXT_MINOR = f"{_MAJOR}.{int(_MINOR) + 1}.0"

REPO = Path(__file__).parents[2]
GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com"]


def write_schema(directory: Path, schema: dict) -> Path:
    path = directory / "schema" / "run.schema.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(schema_text(schema), encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository whose `main` holds the current schema."""
    monkeypatch.chdir(tmp_path)
    for command in (["init", "-q", "-b", "main"], ["add", "-A"]):
        if command[0] == "add":
            write_schema(tmp_path, run_schema())
        subprocess.run([*GIT, *command], cwd=tmp_path, check=True)
    subprocess.run([*GIT, "commit", "-q", "-m", "publish"], cwd=tmp_path, check=True)
    return tmp_path


def edited(edit, *, version=None) -> dict:
    schema = copy.deepcopy(run_schema())
    edit(schema)
    if version is not None:
        schema["properties"]["schema_version"]["default"] = version
    return schema


def check(*extra: str):
    return runner.invoke(app, ["schema", "check", "--base", "main", *extra])


# --- export -----------------------------------------------------------------------


@pytest.mark.unit
def test_export_writes_the_schema_into_the_given_directory(tmp_path):
    result = runner.invoke(app, ["schema", "export", "--out", str(tmp_path / "out")])

    assert result.exit_code == 0
    written = (tmp_path / "out" / "run.schema.json").read_text(encoding="utf-8")
    assert written == schema_text(run_schema())
    assert json.loads(written)["title"] == "Run"
    policy = (tmp_path / "out" / "policy.schema.json").read_text(encoding="utf-8")
    assert policy == schema_text(policy_schema())
    assert json.loads(policy)["title"] == "Policy"


@pytest.mark.unit
def test_check_mode_writes_nothing_and_passes_when_files_are_current(tmp_path):
    args = ["schema", "export", "--out", str(tmp_path), "--check"]
    assert runner.invoke(app, args).exit_code == 1
    assert not (tmp_path / "run.schema.json").exists()

    runner.invoke(app, ["schema", "export", "--out", str(tmp_path)])
    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert "up to date" in result.output


@pytest.mark.unit
def test_check_mode_fails_on_a_stale_schema(tmp_path):
    runner.invoke(app, ["schema", "export", "--out", str(tmp_path)])
    path = tmp_path / "run.schema.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"Run"', '"Old"', 1))

    result = runner.invoke(app, ["schema", "export", "--out", str(tmp_path), "--check"])

    assert result.exit_code == 1
    assert f"out of date: {path}" in result.output
    assert "Old" in path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_types_need_pnpm(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    types = tmp_path / "t.d.ts"

    result = runner.invoke(
        app, ["schema", "export", "--out", str(tmp_path), "--types", str(types)]
    )

    assert result.exit_code == 2
    assert "pnpm" in result.output
    assert not types.exists()


@pytest.mark.integration
@pytest.mark.slow
def test_the_committed_typescript_types_are_what_the_generator_writes():
    if shutil.which("pnpm") is None or not (REPO / "frontend/node_modules").is_dir():
        pytest.skip("needs pnpm and `pnpm install` in frontend/")
    types = REPO / "frontend/src/generated/run.d.ts"

    text = schema_export.generate_typescript(
        schema_text(run_schema()), REPO / "frontend"
    )

    assert text == types.read_text(encoding="utf-8")
    assert "export interface Run {" in text
    assert '"health.status": CheckStatus | null;' in text
    assert 'semantic_type: "numeric";' in text
    assert "Confidence1" not in text


# --- check ------------------------------------------------------------------------


@pytest.mark.unit
def test_an_unchanged_schema_passes(repo):
    result = check()

    assert result.exit_code == 0
    assert "0 change(s), announced" in result.output


@pytest.mark.unit
def test_deleting_a_field_without_a_version_bump_fails(repo):
    def delete_code(schema):
        del schema["$defs"]["Finding"]["properties"]["code"]

    write_schema(repo, edited(delete_code))

    result = check()

    assert result.exit_code == 1
    assert "breaking: #/$defs/Finding/code: property removed" in result.output
    assert "need a minor version bump" in result.output


@pytest.mark.unit
def test_deleting_a_field_with_a_version_bump_passes(repo):
    def delete_code(schema):
        del schema["$defs"]["Finding"]["properties"]["code"]

    write_schema(repo, edited(delete_code, version=NEXT_MINOR))

    result = check()

    assert result.exit_code == 0
    assert "announced" in result.output


@pytest.mark.unit
def test_an_additive_change_needs_a_minor_bump(repo):
    def add_note(schema):
        schema["$defs"]["Finding"]["properties"]["remark"] = {
            "type": "string",
            "default": "",
        }

    write_schema(repo, edited(add_note))
    assert check().exit_code == 1
    write_schema(repo, edited(add_note, version=NEXT_MINOR))

    result = check()

    assert result.exit_code == 0
    assert "additive: #/$defs/Finding/remark" in result.output


@pytest.mark.unit
def test_a_missing_schema_file_is_an_error(repo):
    result = check("--schema", str(Path("schema") / "other.schema.json"))

    assert result.exit_code == 2
    assert "dsbf schema export" in result.output


@pytest.mark.unit
def test_a_base_without_a_schema_passes(repo):
    subprocess.run(["git", "rm", "-q", "-r", "schema"], cwd=repo, check=True)
    subprocess.run([*GIT, "commit", "-q", "-m", "drop"], cwd=repo, check=True)
    write_schema(repo, run_schema())

    result = check()

    assert result.exit_code == 0
    assert "nothing to compare" in result.output


@pytest.mark.unit
def test_an_unknown_base_is_an_error_not_a_pass(repo):
    result = runner.invoke(app, ["schema", "check", "--base", "origin/nowhere"])

    assert result.exit_code == 2
    assert "not a git revision" in result.output
