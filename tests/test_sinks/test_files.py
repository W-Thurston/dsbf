"""``Run.write`` and ``Run.read``: the JSON and NDJSON sinks, and the round trip."""

import json
import warnings
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import dsbf
from dsbf.interfaces import schema_export
from dsbf.model import Finding, Run
from dsbf.sinks import RunReadError, known_sinks
from tests.test_library.frames import build_frame

ROOT = Path(__file__).parents[2]
SCHEMA_DIR = ROOT / "schema"
TITANIC = ROOT / "tests" / "golden" / "data" / "titanic.csv"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def quiet_libraries():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def run() -> Run:
    return dsbf.profile(build_frame(), depth="quick")


@pytest.fixture(scope="module")
def titanic() -> Run:
    return dsbf.profile(TITANIC)


def test_the_registry_lists_json_then_ndjson():
    assert known_sinks()[:2] == ("json", "ndjson")


def test_write_defaults_to_run_json(run, tmp_path):
    written = run.write(tmp_path / "out" / "nested")
    assert written == {"json": tmp_path / "out" / "nested" / "run.json"}
    assert not (tmp_path / "out" / "nested" / "findings.ndjson").exists()


def test_a_run_survives_a_round_trip(run, tmp_path):
    written = run.write(tmp_path)
    assert Run.read(written["json"]) == run
    assert Run.read(tmp_path) == run
    assert dsbf.Run is Run


def test_ndjson_is_one_finding_per_line_in_run_order(run, tmp_path):
    assert run.findings
    path = run.write(tmp_path, sinks=["json", "ndjson"])["ndjson"]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [Finding.model_validate_json(line) for line in lines] == list(run.findings)
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_ndjson_of_a_run_without_findings_is_empty(run, tmp_path):
    results = {k: v.model_copy(update={"findings": ()}) for k, v in run.results.items()}
    quiet = run.model_copy(update={"results": results})
    quiet.__dict__.pop("findings")  # model_copy carries the cached view along
    path = quiet.write(tmp_path, sinks=["ndjson"])["ndjson"]
    assert path.read_text(encoding="utf-8") == ""


def test_an_unknown_sink_writes_nothing(run, tmp_path):
    with pytest.raises(
        ValueError, match="unknown sink 'csv'.*known sinks: json, ndjson"
    ):
        run.write(tmp_path / "out", sinks=["json", "csv"])
    assert not (tmp_path / "out").exists()


def test_a_repeated_sink_is_written_once(run, tmp_path):
    assert list(run.write(tmp_path, sinks=["json", "json"])) == ["json"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "cannot read run"),
        ("{not json", "not a readable run"),
        ("{}", "not a readable run"),
    ],
)
def test_reading_a_bad_file_names_the_file(tmp_path, content, message):
    path = tmp_path / "run.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(RunReadError, match=message) as caught:
        Run.read(path)
    assert str(path) in str(caught.value)


def test_a_run_of_another_major_version_is_refused(run, tmp_path):
    document = json.loads(run.model_dump_json())
    document["schema_version"] = "99.0.0"
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RunReadError, match="schema_version"):
        Run.read(path)


def test_the_titanic_run_validates_against_the_published_schemas(titanic, tmp_path):
    document = json.loads(titanic.write(tmp_path)["json"].read_text(encoding="utf-8"))
    run_schema = json.loads((SCHEMA_DIR / "run.schema.json").read_text("utf-8"))
    Draft202012Validator(run_schema).validate(document)
    validated = 0
    for check_id, result in document["results"].items():
        if result["output"] is None:
            continue
        path = schema_export.check_schema_path(SCHEMA_DIR, check_id)
        schema = json.loads(path.read_text("utf-8"))["output"]
        Draft202012Validator(schema).validate(result["output"])
        validated += 1
    assert validated >= 20


def test_check_params_validate_against_their_schemas(titanic):
    for check_id, result in titanic.results.items():
        path = schema_export.check_schema_path(SCHEMA_DIR, check_id)
        schema = json.loads(path.read_text("utf-8"))["params"]
        Draft202012Validator(schema).validate(result.params)
