import warnings
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

import dsbf
from dsbf.service import create_app
from dsbf.store import SqliteRunRepository
from tests.fixtures.adversarial import build_adversarial_frame
from tests.golden.datasets import KITCHEN5K
from tests.test_engine.test_executor import TOYS, broken
from tests.test_engine.toys import run_plan

TITANIC = Path(__file__).parents[1] / "golden" / "data" / "titanic.csv"
TITANIC_KEY = "/srv/data/titanic.csv"
HOSTILE_COLUMNS = ("a/b", "c#d", "100%", "x y?", "..", "naïve")
LOCAL = "http://127.0.0.1"


@pytest.fixture(scope="package", autouse=True)
def quiet_libraries():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def hostile_frame(n_rows: int = 60) -> pl.DataFrame:
    return pl.DataFrame(
        {
            name: [(i * (position + 3)) % 11 for i in range(n_rows)]
            for position, name in enumerate(HOSTILE_COLUMNS)
        }
    )


@pytest.fixture(scope="package")
def store(tmp_path_factory):
    """A store of titanic (a key with slashes), kitchen5k, a hostile-named frame
    and two runs of one small dataset that differ by planted missing values."""
    root = tmp_path_factory.mktemp("service")
    repository = SqliteRunRepository(root / "dsbf.db", root / "artifacts")
    runs = {
        "titanic": dsbf.profile(TITANIC, dataset=TITANIC_KEY),
        "kitchen5k": dsbf.profile(
            KITCHEN5K.write_csv(root), depth="deep", time_index="normal_date"
        ),
    }
    runs["hostile"] = dsbf.profile(hostile_frame(), dataset="hostile", depth="quick")
    runs["adversarial"] = dsbf.profile(
        build_adversarial_frame(), dataset="adversarial", depth="quick"
    )
    base = pl.DataFrame({"id": range(80), "amount": [10 + i % 5 for i in range(80)]})
    head = base.with_columns(
        pl.when(pl.col("id") % 3 == 0)
        .then(None)
        .otherwise(pl.col("amount"))
        .alias("amount")
    )
    runs["before"] = dsbf.profile(base, dataset="planted", depth="quick")
    runs["after"] = dsbf.profile(head, dataset="planted", depth="quick")
    # A check that raised keeps its traceback as a JSON artifact.
    runs["failed"] = run_plan([broken(TOYS[0]), *TOYS[1:]])
    for run in runs.values():
        repository.save(run)
    nosnap = SqliteRunRepository(root / "none.db", root / "none", snapshot_rows=0)
    runs["no_snapshot"] = dsbf.profile(base, dataset="bare", depth="quick")
    nosnap.save(runs["no_snapshot"])
    yield repository, runs, nosnap
    repository.close()
    nosnap.close()


@pytest.fixture(scope="package")
def runs(store):
    return store[1]


@pytest.fixture(scope="package")
def client(store):
    return TestClient(create_app(store[0]), base_url=LOCAL)


@pytest.fixture(scope="package")
def bare_client(store):
    return TestClient(create_app(store[2]), base_url=LOCAL)
