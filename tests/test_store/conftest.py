import warnings

import pytest

import dsbf
from dsbf.model import Run
from dsbf.store import SqliteRunRepository
from tests.test_library.frames import build_frame


@pytest.fixture(scope="package", autouse=True)
def quiet_libraries():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def repo(tmp_path):
    repository = SqliteRunRepository(tmp_path / "dsbf.db", tmp_path / "artifacts")
    yield repository
    repository.close()


@pytest.fixture(scope="package")
def demo_run() -> Run:
    return dsbf.profile(build_frame(), depth="quick", dataset="demo")


@pytest.fixture(scope="package")
def second_demo_run() -> Run:
    return dsbf.profile(build_frame(), depth="quick", dataset="demo")
