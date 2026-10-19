"""The version comes from the installed package, not from the working directory."""

import subprocess
from importlib.metadata import PackageNotFoundError

import pytest

from dsbf.utils import versioning
from dsbf.utils.versioning import UNKNOWN_VERSION, get_dsbf_version

pytestmark = pytest.mark.unit


def test_the_version_is_the_installed_packages(monkeypatch):
    monkeypatch.setattr(versioning, "version", lambda name: f"{name}-9.8.7")

    assert get_dsbf_version() == "dsbf-9.8.7"


def test_a_missing_package_gives_the_unknown_version(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(versioning, "version", missing)

    assert get_dsbf_version() == UNKNOWN_VERSION


def test_the_version_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    expected = get_dsbf_version()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "tag", "v99.0.0"],
        check=False,
        capture_output=True,
    )
    monkeypatch.chdir(tmp_path)

    assert get_dsbf_version() == expected
    assert expected != UNKNOWN_VERSION
