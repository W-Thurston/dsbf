"""``dsbf profile`` runs ``dsbf.profile()`` and writes ``run.json``."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from dsbf.interfaces.cli import app
from dsbf.model import Run

runner = CliRunner()

ROOT = Path(__file__).parents[2]
SCHEMA = json.loads((ROOT / "schema" / "run.schema.json").read_text(encoding="utf-8"))
TITANIC = ROOT / "tests" / "golden" / "data" / "titanic.csv"

DATA = """\
id,amount,group,note
1,10.5,a,x
2,11.0,b,y
3,,a,z
4,9.5,b,x
5,10.1,a,y
6,10.9,b,z
"""


@pytest.fixture
def csv(tmp_path: Path) -> Path:
    path = tmp_path / "small.csv"
    path.write_text(DATA, encoding="utf-8")
    return path


@pytest.fixture
def dsbf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("DSBF_HOME", str(home))
    return home


def read_run(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator(SCHEMA).validate(document)
    Run.model_validate(document)
    return document


@pytest.mark.integration
class TestWritesTheRun:
    def test_titanic_writes_a_valid_run_and_prints_the_health(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["profile", str(TITANIC), "--out", "out"])

        assert result.exit_code == 0, result.output
        document = read_run(tmp_path / "out" / "run.json")
        assert document["dataset"]["name"] == "titanic"
        assert result.stdout.startswith("DSBF run ")
        assert str(tmp_path / "out" / "run.json") in result.stdout

    def test_a_parquet_file_with_a_datetime_column_is_profiled(self, tmp_path):
        start = datetime(2024, 1, 1)
        path = tmp_path / "events.parquet"
        pl.DataFrame(
            {
                "when": [start + timedelta(days=i) for i in range(60)],
                "value": [float(i % 7) for i in range(60)],
            }
        ).write_parquet(path)

        out = tmp_path / "o"

        result = runner.invoke(app, ["profile", str(path), "--out", str(out)])

        assert result.exit_code == 0, result.output
        document = read_run(tmp_path / "o" / "run.json")
        assert document["dataset"]["n_rows"] == 60

    def test_the_default_directory_is_named_by_the_run_id(self, csv, dsbf_home):
        result = runner.invoke(app, ["profile", str(csv)])

        assert result.exit_code == 0, result.output
        (run_file,) = (dsbf_home / "outputs").glob("*/run.json")
        document = read_run(run_file)
        assert run_file.parent.name == document["run_id"]

    def test_a_relative_out_is_resolved_against_the_working_directory(
        self, csv, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["profile", str(csv), "--out", "a/b"])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "a" / "b" / "run.json").is_file()
        assert str(tmp_path / "a" / "b" / "run.json") in result.stdout

    def test_a_param_reaches_the_check(self, csv, tmp_path):
        out = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "profile",
                str(csv),
                "--out",
                str(out),
                "--param",
                "null_rate.notice_share=0.1",
            ],
        )

        assert result.exit_code == 0, result.output
        params = read_run(out / "run.json")["results"]["null_rate"]["params"]
        assert params["notice_share"] == 0.1

    def test_a_param_wins_over_the_target_setting(self, csv, tmp_path):
        out = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "profile",
                str(csv),
                "--out",
                str(out),
                "--target",
                "group",
                "--param",
                "class_imbalance.target=note",
            ],
        )

        assert result.exit_code == 0, result.output
        results = read_run(out / "run.json")["results"]
        assert results["class_imbalance"]["params"]["target"] == "note"

    def test_depth_limits_the_checks(self, csv, tmp_path):
        counts = {}
        for depth in ("quick", "deep"):
            out = tmp_path / depth
            result = runner.invoke(
                app, ["profile", str(csv), "--out", str(out), "--depth", depth]
            )
            assert result.exit_code == 0, result.output
            counts[depth] = len(read_run(out / "run.json")["results"])

        assert counts["quick"] < counts["deep"]


@pytest.mark.unit
class TestSetupFailures:
    def test_a_missing_file_exits_2_and_writes_nothing(self, tmp_path, dsbf_home):
        result = runner.invoke(app, ["profile", str(tmp_path / "nope.csv")])

        assert result.exit_code == 2
        assert "error:" in result.output
        assert not (dsbf_home / "outputs").exists()

    def test_an_unknown_check_exits_2(self, csv, dsbf_home):
        result = runner.invoke(app, ["profile", str(csv), "--param", "nulls.x=1"])

        assert result.exit_code == 2
        assert "unknown check 'nulls'" in result.output

    def test_an_unknown_parameter_exits_2(self, csv, dsbf_home):
        result = runner.invoke(app, ["profile", str(csv), "--param", "null_rate.x=1"])

        assert result.exit_code == 2
        assert "unknown parameter 'x'" in result.output

    def test_a_target_that_is_not_a_column_exits_2(self, csv, dsbf_home):
        result = runner.invoke(app, ["profile", str(csv), "--target", "price"])

        assert result.exit_code == 2
        assert "'price' is not a column" in result.output
        assert not (dsbf_home / "outputs").exists()

    def test_an_unknown_pack_exits_2(self, csv, dsbf_home):
        result = runner.invoke(app, ["profile", str(csv), "--pack", "nope"])

        assert result.exit_code == 2

    def test_a_malformed_param_is_a_usage_error(self, csv):
        result = runner.invoke(app, ["profile", str(csv), "--param", "oops"])

        assert result.exit_code == 2
        assert "check.key=value" in result.output

    def test_the_legacy_commands_are_gone(self):
        for command in ("run", "quickstart"):
            assert runner.invoke(app, [command]).exit_code == 2
