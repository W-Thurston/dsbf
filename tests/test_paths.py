# tests/test_paths.py
"""The runtime locations resolve from DSBF_HOME and the per-user data directory."""

from pathlib import Path

import pytest

from dsbf import paths

pytestmark = pytest.mark.unit


def test_home_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DSBF_HOME", str(tmp_path / "custom"))

    assert paths.dsbf_home() == (tmp_path / "custom").resolve()
    assert paths.default_outputs_dir() == paths.dsbf_home() / "outputs"


def test_home_expands_the_user_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DSBF_HOME", "~/dsbf-home")

    assert paths.dsbf_home() == (tmp_path / "dsbf-home").resolve()


def test_home_defaults_to_the_per_user_data_directory(monkeypatch):
    monkeypatch.delenv("DSBF_HOME", raising=False)
    monkeypatch.setattr(paths, "user_data_dir", lambda name: f"/data/{name}")

    assert paths.dsbf_home() == Path("/data/dsbf").resolve()


def test_home_is_read_on_every_call_and_never_created(monkeypatch, tmp_path):
    target = tmp_path / "later"
    monkeypatch.setenv("DSBF_HOME", str(target))

    assert paths.dsbf_home() == target.resolve()
    assert not target.exists()
