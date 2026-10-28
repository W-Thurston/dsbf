"""``--sink store`` and ``dsbf runs``, end to end under a temporary DSBF home."""

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dsbf.interfaces.cli import app

pytestmark = pytest.mark.integration

runner = CliRunner()

DATA = "id,amount,group\n" + "".join(
    f"{i},{10 + i % 5},{'ab'[i % 2]}\n" for i in range(1, 60)
)
CHANGED = DATA + "".join(f"{i},,{'ab'[i % 2]}\n" for i in range(60, 100))


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("DSBF_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def store(path: Path) -> None:
    result = runner.invoke(
        app, ["check", str(path), "--sink", "store", "--dataset", "d"]
    )
    assert result.exit_code == 0, result.output


def listed() -> list[str]:
    result = runner.invoke(app, ["runs", "list", "--json"])
    assert result.exit_code == 0, result.output
    return [item["run_id"] for item in json.loads(result.output)["items"]]


def test_a_stored_run_is_listed_and_writes_no_directory(home, tmp_path):
    store(write(tmp_path, "d.csv", DATA))
    assert len(listed()) == 1
    assert not (home / "outputs").exists()
    assert (home / "dsbf.db").exists()


def test_store_beside_another_sink_writes_both(home, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "check",
            str(write(tmp_path, "d.csv", DATA)),
            "-s",
            "store",
            "-s",
            "json",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "run.json").exists()
    assert len(listed()) == 1


def test_diff_reports_a_planted_change(home, tmp_path):
    store(write(tmp_path, "d.csv", DATA))
    store(write(tmp_path, "d.csv", CHANGED))
    newest, oldest = listed()
    result = runner.invoke(app, ["runs", "diff", oldest, newest, "--json"])
    assert result.exit_code == 0, result.output
    diff = json.loads(result.output)
    assert diff["same_dataset"]
    assert diff["new"] or diff["changed"]


def test_show_prints_the_health_summary(home, tmp_path):
    store(write(tmp_path, "d.csv", DATA))
    (run_id,) = listed()
    result = runner.invoke(app, ["runs", "show", run_id])
    assert result.exit_code == 0, result.output
    assert "DSBF run" in result.output


def test_delete_removes_the_run_and_its_files(home, tmp_path):
    store(write(tmp_path, "d.csv", DATA))
    (run_id,) = listed()
    assert runner.invoke(app, ["runs", "delete", run_id]).exit_code == 0
    assert listed() == []
    assert not (home / "artifacts" / run_id).exists()


def test_deleting_an_unknown_run_is_a_setup_error(home):
    result = runner.invoke(
        app, ["runs", "delete", "00000000-0000-0000-0000-000000000000"]
    )
    assert result.exit_code == 2


def test_gc_keeps_the_newest_run(home, tmp_path):
    store(write(tmp_path, "d.csv", DATA))
    store(write(tmp_path, "d.csv", CHANGED))
    newest, _ = listed()
    result = runner.invoke(app, ["runs", "gc", "--keep", "1"])
    assert result.exit_code == 0, result.output
    assert listed() == [newest]
    assert [p.name for p in (home / "artifacts").iterdir()] == [newest]


def test_import_stores_a_run_the_json_sink_wrote(home, tmp_path):
    out = tmp_path / "ci"
    data = write(tmp_path, "d.csv", DATA)
    assert runner.invoke(app, ["check", str(data), "-o", str(out)]).exit_code == 0
    result = runner.invoke(app, ["runs", "import", str(out)])
    assert result.exit_code == 0, result.output
    assert len(listed()) == 1
    again = runner.invoke(app, ["runs", "import", str(out)])
    assert again.exit_code == 2


def test_diff_reads_run_json_paths_without_a_store(home, tmp_path):
    data = write(tmp_path, "d.csv", DATA)
    for name in ("a", "b"):
        assert (
            runner.invoke(
                app, ["check", str(data), "-o", str(tmp_path / name)]
            ).exit_code
            == 0
        )
    result = runner.invoke(
        app, ["runs", "diff", str(tmp_path / "a"), str(tmp_path / "b")]
    )
    assert result.exit_code == 0, result.output
    assert "0 new, 0 resolved" in result.output


def test_an_unknown_reference_is_a_setup_error(home):
    result = runner.invoke(app, ["runs", "show", "nonsense"])
    assert result.exit_code == 2


def test_the_store_sink_without_the_extra_is_a_setup_error(home, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "dsbf.store", None)
    path = write(tmp_path, "d.csv", DATA)
    result = runner.invoke(app, ["check", str(path), "--sink", "store"])
    assert result.exit_code == 2
    assert not (home / "outputs").exists()
