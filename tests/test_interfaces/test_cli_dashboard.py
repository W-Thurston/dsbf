"""``dsbf dashboard`` serves the API over the store, on this machine only."""

import pytest
import uvicorn
from typer.testing import CliRunner

from dsbf.interfaces.cli import app

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture
def launched(monkeypatch, tmp_path):
    monkeypatch.setenv("DSBF_HOME", str(tmp_path / "home"))
    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda application, **kw: calls.append(kw))
    return calls


def test_it_serves_on_the_given_local_address(launched):
    result = runner.invoke(app, ["dashboard", "--port", "9123", "--no-open"])

    assert result.exit_code == 0, result.output
    assert launched[0]["host"] == "127.0.0.1" and launched[0]["port"] == 9123
    assert "http://127.0.0.1:9123/" in result.output


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com"])  # noqa: S104
def test_it_refuses_an_address_beyond_this_machine(launched, host):
    result = runner.invoke(app, ["dashboard", "--host", host, "--no-open"])

    assert result.exit_code == 2
    assert "not a local address" in result.output
    assert launched == []


def test_it_opens_the_browser_unless_told_not_to(launched, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    monkeypatch.setattr(
        "threading.Timer",
        lambda delay, fn, args: type("T", (), {"start": lambda self: fn(*args)})(),
    )

    runner.invoke(app, ["dashboard", "--port", "9124"])

    assert opened == ["http://127.0.0.1:9124/"]
