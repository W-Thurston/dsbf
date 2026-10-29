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
NEXT_MAJOR = f"{int(_MAJOR) + 1}.0.0"

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
    assert "need a major version bump" in result.output


@pytest.mark.unit
def test_deleting_a_field_with_a_version_bump_passes(repo):
    def delete_code(schema):
        del schema["$defs"]["Finding"]["properties"]["code"]

    write_schema(repo, edited(delete_code, version=NEXT_MAJOR))

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


# --- check schemas -----------------------------------------------------------------


@pytest.mark.unit
def test_export_writes_one_schema_per_core_check(tmp_path):
    result = runner.invoke(app, ["schema", "export", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    written = sorted((tmp_path / "checks").glob("*.schema.json"))
    assert len(written) == len(schema_export.check_schemas())
    document = json.loads((tmp_path / "checks" / "null_rate.schema.json").read_text())
    assert document["check_id"] == "null_rate"
    assert set(document) >= {"check_version", "params", "output"}


@pytest.mark.unit
def test_the_committed_check_schemas_are_current():
    result = runner.invoke(
        app, ["schema", "export", "--out", str(REPO / "schema"), "--check"]
    )
    assert result.exit_code == 0, result.output


@pytest.mark.unit
def test_check_mode_flags_a_schema_file_with_no_check(tmp_path):
    runner.invoke(app, ["schema", "export", "--out", str(tmp_path)])
    (tmp_path / "checks" / "gone.schema.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(app, ["schema", "export", "--out", str(tmp_path), "--check"])
    assert result.exit_code == 1
    assert "gone.schema.json" in result.output
    runner.invoke(app, ["schema", "export", "--out", str(tmp_path)])
    assert not (tmp_path / "checks" / "gone.schema.json").exists()


@pytest.fixture
def checks_repo(repo):
    """The `repo` fixture's main, with the check schemas published too."""
    runner.invoke(app, ["schema", "export", "--out", str(repo / "schema")])
    for command in (["add", "-A"], ["commit", "-q", "-m", "checks"]):
        subprocess.run([*GIT, *command], cwd=repo, check=True)
    return repo


def edit_check(repo: Path, check_id: str, edit, *, version=None) -> None:
    path = repo / "schema" / "checks" / f"{check_id}.schema.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    if version is not None:
        document["check_version"] = version
    path.write_text(schema_text(document), encoding="utf-8")


def drop_output_property(document: dict) -> None:
    properties = document["output"]["properties"]
    properties.pop(next(iter(properties)))


@pytest.mark.unit
def test_unchanged_check_schemas_pass(checks_repo):
    assert check().exit_code == 0


@pytest.mark.unit
def test_a_breaking_output_change_without_a_version_bump_fails(checks_repo):
    edit_check(checks_repo, "null_rate", drop_output_property)
    result = check()
    assert result.exit_code == 1
    assert "null_rate" in result.output


@pytest.mark.unit
def test_a_breaking_output_change_with_a_major_bump_passes(checks_repo):
    edit_check(checks_repo, "null_rate", drop_output_property, version="99.0.0")
    assert check().exit_code == 0


@pytest.mark.unit
def test_removing_a_check_schema_fails(checks_repo):
    (checks_repo / "schema" / "checks" / "null_rate.schema.json").unlink()
    result = check()
    assert result.exit_code == 1
    assert "null_rate" in result.output


@pytest.fixture
def tables_repo(checks_repo):
    """The `checks_repo` fixture's main, with the Parquet column listing published."""
    return checks_repo


def edit_tables(repo: Path, edit, *, version=None) -> None:
    path = repo / "schema" / "tables.schema.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    if version is not None:
        document["schema_version"] = version
    path.write_text(schema_text(document), encoding="utf-8")


def drop_column(document: dict) -> None:
    document["tables"]["findings"]["columns"].pop()


@pytest.mark.unit
def test_unchanged_table_schemas_pass(tables_repo):
    assert check().exit_code == 0


@pytest.mark.unit
def test_dropping_a_column_without_a_version_bump_fails(tables_repo):
    edit_tables(tables_repo, drop_column)
    result = check()
    assert result.exit_code == 1
    assert "findings.non_finite_metrics" in result.output


@pytest.mark.unit
def test_dropping_a_column_with_a_major_bump_passes(tables_repo):
    edit_tables(tables_repo, drop_column, version="2.0.0")
    assert check().exit_code == 0


@pytest.mark.unit
def test_removing_the_table_listing_fails(tables_repo):
    (tables_repo / "schema" / "tables.schema.json").unlink()
    assert check().exit_code == 2


# --- policy schema -----------------------------------------------------------------


def edit_policy(repo: Path, edit, *, version=None) -> None:
    path = repo / "schema" / "policy.schema.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    if version is not None:
        document["properties"]["version"]["default"] = version
        document["properties"]["version"]["const"] = version
    path.write_text(schema_text(document), encoding="utf-8")


def drop_rules(document: dict) -> None:
    del document["properties"]["rules"]


def add_option(document: dict) -> None:
    document["properties"]["note"] = {"type": "string", "default": ""}


@pytest.mark.unit
def test_unchanged_policy_schema_passes(tables_repo):
    assert check().exit_code == 0


@pytest.mark.unit
def test_dropping_a_policy_field_without_a_version_bump_fails(tables_repo):
    edit_policy(tables_repo, drop_rules)

    result = check()

    assert result.exit_code == 1
    assert "policy: breaking" in result.output
    assert "Policy.version" in result.output


@pytest.mark.unit
def test_dropping_a_policy_field_with_a_new_version_passes(tables_repo):
    edit_policy(tables_repo, drop_rules, version=2)
    assert check().exit_code == 0


@pytest.mark.unit
def test_an_additive_policy_change_needs_no_version_bump(tables_repo):
    edit_policy(tables_repo, add_option)

    result = check()

    assert result.exit_code == 0
    assert "policy: additive" in result.output


@pytest.mark.unit
def test_the_rank_tables_list_each_vocabulary_lowest_first():
    text = schema_export.ranks_module()

    assert 'SEVERITY_ORDER = ["info", "notice", "warn", "error"] as const' in text
    assert 'CONFIDENCE_ORDER = ["low", "moderate", "high"] as const' in text
    assert "export function severityRank(" in text


@pytest.mark.integration
@pytest.mark.slow
def test_the_committed_api_types_and_ranks_are_what_the_generators_write():
    if shutil.which("pnpm") is None or not (REPO / "frontend/node_modules").is_dir():
        pytest.skip("needs pnpm and `pnpm install` in frontend/")
    generated = REPO / "frontend/src/generated"

    types = schema_export.generate_api_types(
        (REPO / "schema/openapi.json").read_text(encoding="utf-8"), REPO / "frontend"
    )

    assert types == (generated / schema_export.API_TYPES_FILE).read_text("utf-8")
    assert '"/v2/jobs"' in types
    assert schema_export.ranks_module() == (
        generated / schema_export.RANKS_FILE
    ).read_text("utf-8")


# --- run comparison schema ---------------------------------------------------------


def edit_diff(repo: Path, edit, *, version=None) -> None:
    path = repo / "schema" / "diff.schema.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    if version is not None:
        document["properties"]["schema_version"]["default"] = version
    path.write_text(schema_text(document), encoding="utf-8")


def drop_tolerance(document: dict) -> None:
    del document["properties"]["tolerance"]


@pytest.mark.unit
def test_unchanged_diff_schema_passes(checks_repo):
    assert check().exit_code == 0


@pytest.mark.unit
def test_dropping_a_diff_field_without_a_major_bump_fails(checks_repo):
    edit_diff(checks_repo, drop_tolerance)
    result = check()
    assert result.exit_code == 1
    assert "diff: breaking" in result.output
    assert "DIFF_SCHEMA_VERSION" in result.output
    edit_diff(checks_repo, lambda document: None, version="1.1.0")
    assert check().exit_code == 1


@pytest.mark.unit
def test_dropping_a_diff_field_with_a_major_bump_passes(checks_repo):
    edit_diff(checks_repo, drop_tolerance, version="2.0.0")
    assert check().exit_code == 0
