"""Every `python` and `bash` block in a guide runs, in order, on a sample table."""

import os
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dsbf.interfaces.cli import app
from tests.test_docs.conftest import write_customers

ROOT = Path(__file__).parents[2]
GUIDES = sorted((ROOT / "docs" / "guides").glob("*.md"))
BLOCK = re.compile(r"^```(python|bash)\n(.*?)^```", re.S | re.M)
# A command that is meant to end with this exit code says so in a trailing comment.
EXIT = re.compile(r"#\s*exits (\d)\s*$")

pytestmark = pytest.mark.integration


def blocks(guide: Path) -> list[tuple[str, str]]:
    return BLOCK.findall(guide.read_text(encoding="utf-8"))


def test_the_guides_exist_and_hold_code():
    assert {g.stem for g in GUIDES} == {
        "compatibility",
        "dashboard",
        "pipeline",
        "policy",
        "standalone-local",
        "write-a-check",
    }
    assert all(blocks(guide) for guide in GUIDES)


@pytest.mark.parametrize("guide", GUIDES, ids=lambda g: g.stem)
def test_every_block_runs(guide, tmp_path, monkeypatch):
    write_customers(tmp_path / "data.csv")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DSBF_HOME", str(tmp_path / "home"))
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [n for n in sys.modules if n.split(".")[0] in {"my_checks"}]:
        del sys.modules[name]
    namespace: dict[str, object] = {"__name__": "__guide__"}
    runner = CliRunner()
    for language, code in blocks(guide):
        if language == "python":
            exec(compile(code, str(guide), "exec"), namespace)  # noqa: S102
            continue
        for line in code.splitlines():
            if not line.strip():
                continue
            expected = EXIT.search(line)
            command = EXIT.sub("", line).split()
            assert command[0] == "dsbf", line
            result = runner.invoke(app, command[1:])
            want = int(expected.group(1)) if expected else 0
            assert result.exit_code == want, f"{line}\n{result.output}"
    assert os.getcwd() == str(tmp_path)
