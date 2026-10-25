# tests/test_hermetic_runtime.py
"""Guards for the session-wide isolation set up in ``tests/conftest.py``."""

from pathlib import Path

import pytest

import dsbf
from dsbf.paths import default_outputs_dir, dsbf_home

pytestmark = pytest.mark.unit

REPO_ROOT = Path(dsbf.__file__).resolve().parents[2]


def test_session_runs_from_a_temporary_directory(isolated_working_directory):
    cwd = Path.cwd().resolve()

    assert cwd == isolated_working_directory
    assert cwd != REPO_ROOT
    assert not cwd.is_relative_to(REPO_ROOT)


def test_home_is_isolated_from_the_user_directory():
    home = dsbf_home()

    assert not home.is_relative_to(REPO_ROOT)
    assert home.parent == Path.cwd().resolve().parent


def test_default_outputs_land_under_the_temporary_home():
    assert default_outputs_dir().is_relative_to(dsbf_home())
