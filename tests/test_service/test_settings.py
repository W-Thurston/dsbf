"""A dataset's saved settings, and rerunning it with them."""

import shutil

import pytest
from fastapi.testclient import TestClient

from dsbf.service import create_app
from dsbf.store import SqliteRunRepository
from tests.test_service.conftest import LOCAL, TITANIC
from tests.test_service.test_jobs import QUICK, start, wait

pytestmark = pytest.mark.integration

KEY = "titanic-settings"


@pytest.fixture
def repo(tmp_path):
    repository = SqliteRunRepository(tmp_path / "dsbf.db", tmp_path / "artifacts")
    yield repository
    repository.close()


@pytest.fixture
def client(repo):
    with TestClient(create_app(repo), base_url=LOCAL) as client:
        yield client


@pytest.fixture
def source(tmp_path):
    return shutil.copy(TITANIC, tmp_path / "titanic.csv")


def first_run(client, source):
    job = start(client, source, settings={**QUICK, "dataset": KEY})
    return wait(client, job["job_id"])


def status_of(client, run_id, check_id):
    return client.get(f"/v2/runs/{run_id}/checks/{check_id}").json()["status"]


class TestOverlay:
    def test_a_dataset_without_saved_settings_has_an_empty_overlay(
        self, client, source
    ):
        first_run(client, source)
        assert client.get(f"/v2/datasets/{KEY}/settings").json() == {}

    def test_saving_keeps_only_the_fields_set_and_replaces_the_earlier_overlay(
        self, client, source
    ):
        first_run(client, source)
        url = f"/v2/datasets/{KEY}/settings"
        assert client.put(
            url, json={"target": "survived", "depth": "quick"}
        ).json() == {
            "target": "survived",
            "depth": "quick",
        }
        assert client.put(url, json={"time_index": "age"}).status_code == 200
        assert client.get(url).json() == {"time_index": "age"}

    def test_deleting_forgets_the_overlay(self, client, source):
        first_run(client, source)
        url = f"/v2/datasets/{KEY}/settings"
        client.put(url, json={"target": "survived"})
        assert client.delete(url).status_code == 204
        assert client.get(url).json() == {}

    @pytest.mark.parametrize(
        "body", [{"depth": "bogus"}, {"no_such_setting": 1}, {"dataset": "other"}]
    )
    def test_settings_the_profile_would_reject_are_refused(self, client, source, body):
        first_run(client, source)
        response = client.put(f"/v2/datasets/{KEY}/settings", json=body)
        assert response.status_code == 422
        assert client.get(f"/v2/datasets/{KEY}/settings").json() == {}

    def test_a_dataset_with_no_runs_has_no_settings(self, client):
        assert client.get("/v2/datasets/nothing/settings").status_code == 404
        assert client.put("/v2/datasets/nothing/settings", json={}).status_code == 404
        assert client.post("/v2/datasets/nothing/rerun").status_code == 404


class TestRerun:
    def test_a_target_in_the_overlay_makes_class_imbalance_run(self, client, source):
        before = first_run(client, source)
        status = status_of(client, before["run_id"], "class_imbalance")
        assert status == "not_applicable"

        client.put(f"/v2/datasets/{KEY}/settings", json={"target": "survived"})
        response = client.post(f"/v2/datasets/{KEY}/rerun")
        assert response.status_code == 202
        after = wait(client, response.json()["job_id"])

        assert after["state"] == "succeeded", after["error"]
        assert after["run_id"] != before["run_id"]
        assert status_of(client, after["run_id"], "class_imbalance") == "ok"
        listed = client.get(f"/v2/datasets/{KEY}/runs").json()
        assert listed["total"] == 2

    def test_a_source_that_no_longer_exists_fails_the_job_with_the_reason(
        self, client, source, tmp_path
    ):
        first_run(client, source)
        (tmp_path / "titanic.csv").unlink()

        job = client.post(f"/v2/datasets/{KEY}/rerun").json()
        done = wait(client, job["job_id"])

        assert done["state"] == "failed"
        assert done["run_id"] is None
        assert "titanic.csv" in done["error"]
