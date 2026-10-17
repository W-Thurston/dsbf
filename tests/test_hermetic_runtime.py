# tests/test_hermetic_runtime.py
"""Guards for the session-wide isolation set up in ``tests/conftest.py``."""

from pathlib import Path

import pytest

import dsbf
from dsbf.config import load_default_config
from dsbf.core.base_engine import BaseEngine
from dsbf.paths import default_db_path, dsbf_home
from dsbf.storage.schema import _resolve_db_path

pytestmark = pytest.mark.unit

REPO_ROOT = Path(dsbf.__file__).resolve().parents[2]


class _NoopEngine(BaseEngine):
    def run(self):
        pass


def test_session_runs_from_a_temporary_directory(isolated_working_directory):
    cwd = Path.cwd().resolve()

    assert cwd == isolated_working_directory
    assert cwd != REPO_ROOT
    assert not cwd.is_relative_to(REPO_ROOT)


def test_home_is_isolated_from_the_user_directory():
    home = dsbf_home()

    assert not home.is_relative_to(REPO_ROOT)
    assert home.parent == Path.cwd().resolve().parent


def test_database_path_is_isolated_from_the_default():
    db_path = _resolve_db_path()

    assert db_path != default_db_path()
    assert not db_path.resolve().is_relative_to(REPO_ROOT)
    assert db_path.parent == Path.cwd().resolve().parent


def test_default_outputs_land_under_the_temporary_home():
    output_dir = Path(_NoopEngine(config={}).output_dir).resolve()

    assert output_dir.is_relative_to(dsbf_home())


def test_default_config_loads_without_the_repo_as_cwd():
    assert not Path("src/dsbf/config/default_config.yaml").exists()
    assert "tasks" in load_default_config()
