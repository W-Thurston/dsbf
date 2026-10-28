"""Jobs profile a local file into the store, one at a time, and report progress."""

import threading
import time
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from dsbf.engine import RunCancelled
from dsbf.service import create_app
from dsbf.store import SqliteRunRepository
from tests.test_service.conftest import LOCAL, TITANIC
from tests.test_service.test_endpoints import PROBLEM

pytestmark = pytest.mark.integration

QUICK = {"depth": "quick", "packs": ["core"]}


@pytest.fixture
def repo(tmp_path):
    repository = SqliteRunRepository(tmp_path / "dsbf.db", tmp_path / "artifacts")
    yield repository
    repository.close()


@pytest.fixture
def client(repo):
    with TestClient(create_app(repo), base_url=LOCAL) as client:
        yield client


def start(client, source=TITANIC, **body):
    response = client.post("/v2/jobs", json={"source": str(source), **body})
    assert response.status_code == 202, response.text
    return response.json()


def wait(client, job_id, *, states=("succeeded", "failed", "cancelled")):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        job = client.get(f"/v2/jobs/{job_id}").json()["job"]
        if job["state"] in states:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach {states}")


def all_events(client, job_id):
    return client.get(f"/v2/jobs/{job_id}").json()["events"]


class TestProfiling:
    def test_a_job_profiles_titanic_and_its_run_is_in_the_store(self, client):
        job = start(
            client,
            name="titanic",
            settings={**QUICK, "dataset": "titanic-job"},
        )
        assert job["state"] in {"queued", "running"}

        done = wait(client, job["job_id"])

        assert done["state"] == "succeeded"
        assert done["error"] is None
        run = client.get(f"/v2/runs/{done['run_id']}").json()
        assert run["header"]["dataset"]["key"] == "titanic-job"
        assert run["header"]["dataset"]["name"] == "titanic"
        listed = client.get("/v2/runs").json()
        assert [r["run_id"] for r in listed["items"]] == [done["run_id"]]
        assert any(a["ref"] == "snapshot" for a in run["artifacts"])

    def test_events_arrive_in_order_and_end_with_the_run_id(self, client):
        job = start(client, settings=QUICK)
        done = wait(client, job["job_id"])

        events = all_events(client, job["job_id"])

        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
        assert events[-1]["kind"] == "job_finished"
        assert events[-1]["run_id"] == done["run_id"]
        assert events[-1]["state"] == "succeeded"
        started = [e["check_id"] for e in events if e["kind"] == "check_started"]
        finished = [e["check_id"] for e in events if e["kind"] == "check_finished"]
        assert started
        assert sorted(started) == sorted(finished)
        assert all(
            e["total"] == events[0]["total"] for e in events if e["total"] is not None
        )

    def test_events_are_read_after_a_cursor(self, client):
        job = start(client, settings=QUICK)
        wait(client, job["job_id"])

        page = client.get(f"/v2/jobs/{job['job_id']}", params={"limit": 3}).json()
        rest = client.get(
            f"/v2/jobs/{job['job_id']}", params={"after": page["next_after"]}
        ).json()

        assert [e["seq"] for e in page["events"]] == [1, 2, 3]
        assert rest["events"][0]["seq"] == 4
        beyond = client.get(f"/v2/jobs/{job['job_id']}", params={"after": 10_000})
        assert beyond.json()["events"] == []
        assert beyond.json()["next_after"] == 10_000

    def test_the_snapshot_cap_in_the_settings_is_used(self, client):
        job = start(client, settings={**QUICK, "store": {"snapshot_rows": 0}})
        done = wait(client, job["job_id"])

        run = client.get(f"/v2/runs/{done['run_id']}").json()

        assert all(a["ref"] != "snapshot" for a in run["artifacts"])


class TestFailure:
    def test_a_missing_file_fails_the_job_with_the_message_the_cli_prints(
        self, client, tmp_path
    ):
        missing = tmp_path / "nope.csv"
        job = start(client, missing)

        done = wait(client, job["job_id"])

        assert done["state"] == "failed"
        assert done["run_id"] is None
        assert str(missing) in done["error"] or "nope.csv" in done["error"]
        assert all_events(client, job["job_id"])[-1]["message"] == done["error"]

    def test_the_text_is_what_dsbf_check_prints(self, client, tmp_path):
        from typer.testing import CliRunner  # noqa: PLC0415

        from dsbf.interfaces.cli import app  # noqa: PLC0415

        missing = tmp_path / "nope.csv"
        printed = CliRunner().invoke(app, ["check", str(missing)]).output.strip()
        job = start(client, missing)

        done = wait(client, job["job_id"])

        assert printed == f"error: {done['error']}"

    def test_a_check_that_is_not_in_the_plan_fails_with_each_problem(self, client):
        job = start(client, settings={"include": ["no_such_check"]})

        done = wait(client, job["job_id"])

        assert done["state"] == "failed"
        assert "no_such_check" in done["error"]

    def test_a_target_that_is_not_a_column_fails_the_job(self, client):
        job = start(client, settings={"target": "no_such_column"})

        done = wait(client, job["job_id"])

        assert done["state"] == "failed"
        assert "no_such_column" in done["error"]

    def test_a_remote_source_fails_the_job(self, client):
        job = start(client, "https://example.com/data.csv")

        done = wait(client, job["job_id"])

        assert done["state"] == "failed"
        assert "Remote source" in done["error"]


class TestQueue:
    def test_two_jobs_run_one_after_the_other_in_the_order_given(self, client):
        first = start(client, settings=QUICK)
        second = start(client, settings=QUICK)

        a = wait(client, first["job_id"])
        b = wait(client, second["job_id"])

        assert a["finished_at"] <= b["started_at"]
        listed = client.get("/v2/jobs").json()
        assert [j["job_id"] for j in listed["items"]] == [
            second["job_id"],
            first["job_id"],
        ]
        assert listed["total"] == 2
        only = client.get("/v2/jobs", params={"state": "succeeded"}).json()
        assert only["total"] == 2
        assert client.get("/v2/jobs", params={"state": "failed"}).json()["total"] == 0


@pytest.fixture
def held(monkeypatch):
    """Profiling that waits until the job is cancelled, as a long run would."""
    entered = threading.Event()

    def wait_for_cancel(source, settings, *, cancel, **_):
        entered.set()
        assert cancel.wait(60), "the job was never cancelled"
        raise RunCancelled

    monkeypatch.setattr("dsbf.service.jobs.profile_with", wait_for_cancel)
    return entered


class TestCancel:
    def test_a_queued_job_is_cancelled_at_once_and_never_runs(self, client, held):
        blocker = start(client)
        assert held.wait(10)
        waiting = start(client, settings=QUICK)

        response = client.post(f"/v2/jobs/{waiting['job_id']}/cancel")

        assert response.status_code == 200
        assert response.json()["state"] == "cancelled"
        client.post(f"/v2/jobs/{blocker['job_id']}/cancel")
        wait(client, blocker["job_id"])
        job = client.get(f"/v2/jobs/{waiting['job_id']}").json()["job"]
        assert job["state"] == "cancelled"
        assert job["started_at"] is None
        assert job["run_id"] is None

    def test_a_running_job_stops_and_stores_no_run(self, client, held):
        job = start(client)
        assert held.wait(10)

        response = client.post(f"/v2/jobs/{job['job_id']}/cancel")

        assert response.status_code == 200
        assert response.json()["state"] == "running"
        done = wait(client, job["job_id"])
        assert done["state"] == "cancelled"
        assert done["run_id"] is None
        assert client.get("/v2/runs").json()["total"] == 0
        assert all_events(client, job["job_id"])[-1]["state"] == "cancelled"

    def test_a_real_run_cancelled_between_checks_is_not_stored(self, client):
        job = start(client, settings={"depth": "deep", "workers": 1})
        wait(client, job["job_id"], states=("running",))

        client.post(f"/v2/jobs/{job['job_id']}/cancel")

        done = wait(client, job["job_id"])
        assert done["state"] in {"cancelled", "succeeded"}
        assert (done["run_id"] is None) == (done["state"] == "cancelled")
        assert client.get("/v2/runs").json()["total"] == (
            1 if done["state"] == "succeeded" else 0
        )

    def test_a_finished_job_cannot_be_cancelled(self, client):
        job = start(client, settings=QUICK)
        wait(client, job["job_id"])

        response = client.post(f"/v2/jobs/{job['job_id']}/cancel")

        assert response.status_code == 409
        assert response.headers["content-type"] == PROBLEM
        assert "succeeded" in response.json()["detail"]


class TestRestart:
    def test_a_job_left_unfinished_by_a_stopped_service_shows_as_interrupted(
        self, repo
    ):
        from dsbf.store import JobRequest  # noqa: PLC0415

        stale = repo.create_job(JobRequest(source=str(TITANIC)))
        repo.start_job(stale.job_id)

        with TestClient(create_app(repo), base_url=LOCAL) as client:
            job = client.get(f"/v2/jobs/{stale.job_id}").json()["job"]

        assert job["state"] == "interrupted"
        assert job["finished_at"] is not None


class TestRequests:
    def test_an_unknown_job_is_a_404_that_names_it(self, client):
        for response in (
            client.get("/v2/jobs/4c1e0f2e-0000-4000-8000-000000000000"),
            client.post("/v2/jobs/4c1e0f2e-0000-4000-8000-000000000000/cancel"),
        ):
            assert response.status_code == 404
            assert response.headers["content-type"] == PROBLEM
            assert "4c1e0f2e-0000-4000-8000-000000000000" in response.json()["detail"]

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"source": ""},
            {"source": "a.csv", "settings": {"depth": "enormous"}},
            {"source": "a.csv", "settings": {"nonsense": 1}},
            {"source": "a.csv", "settings": {"include": ["x"], "exclude": ["x"]}},
            {"source": "a.csv", "extra": 1},
        ],
    )
    def test_an_invalid_request_is_a_422_and_queues_nothing(self, client, body):
        response = client.post("/v2/jobs", json=body)

        assert response.status_code == 422
        assert response.headers["content-type"] == PROBLEM
        assert response.json()["errors"]
        assert client.get("/v2/jobs").json()["total"] == 0

    @pytest.mark.parametrize(
        "params",
        [{"limit": 0}, {"limit": 501}, {"offset": -1}, {"state": "paused"}],
    )
    def test_the_list_validates_its_parameters(self, client, params):
        assert client.get("/v2/jobs", params=params).status_code == 422

    def test_a_job_id_that_is_not_a_uuid_is_a_422(self, client):
        assert client.get("/v2/jobs/not-a-uuid").status_code == 422
        assert (
            client.get(f"/v2/jobs/{UUID(int=1)}", params={"after": -1}).status_code
            == 422
        )


class TestCrossOriginWrites:
    def test_a_form_post_from_another_site_is_refused_and_queues_nothing(self, client):
        response = client.post(
            "/v2/jobs",
            data={"source": str(TITANIC)},
            headers={"Origin": "https://evil.example"},
        )

        assert response.status_code == 403
        assert response.headers["content-type"] == PROBLEM
        assert client.get("/v2/jobs").json()["total"] == 0

    def test_a_form_post_without_an_origin_is_not_json(self, client):
        response = client.post("/v2/jobs", data={"source": str(TITANIC)})

        assert response.status_code == 415
        assert client.get("/v2/jobs").json()["total"] == 0

    def test_a_json_body_sent_as_text_is_refused(self, client):
        response = client.post(
            "/v2/jobs",
            content='{"source": "x.csv"}',
            headers={"Content-Type": "text/plain"},
        )

        assert response.status_code == 415

    def test_json_from_another_site_is_refused_even_with_the_right_type(self, client):
        response = client.post(
            "/v2/jobs",
            json={"source": str(TITANIC)},
            headers={"Origin": "https://evil.example"},
        )

        assert response.status_code == 403

    def test_a_cancel_from_another_site_is_refused(self, client):
        job = start(client, settings=QUICK)

        response = client.post(
            f"/v2/jobs/{job['job_id']}/cancel", headers={"Origin": "null"}
        )

        assert response.status_code == 403
        wait(client, job["job_id"])

    @pytest.mark.parametrize(
        "origin", ["http://127.0.0.1:8765", "http://localhost:5173", "http://[::1]:80"]
    )
    def test_a_local_page_may_write(self, client, origin):
        job = client.post(
            "/v2/jobs",
            json={"source": str(TITANIC), "settings": QUICK},
            headers={"Origin": origin},
        )

        assert job.status_code == 202
        wait(client, job.json()["job_id"])

    def test_a_page_the_caller_names_may_write(self, repo):
        app = create_app(repo, cors_origins=["https://dash.example"])
        with TestClient(app, base_url=LOCAL) as client:
            response = client.post(
                "/v2/jobs",
                json={"source": str(TITANIC), "settings": QUICK},
                headers={"Origin": "https://dash.example"},
            )
            assert response.status_code == 202
            wait(client, response.json()["job_id"])

    def test_no_cors_headers_are_sent_unless_configured(self, client):
        response = client.options(
            "/v2/jobs",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert "access-control-allow-origin" not in response.headers


def test_the_runner_stops_with_the_app(repo):
    app = create_app(repo)
    with TestClient(app, base_url=LOCAL):
        thread = app.state.runner._thread  # noqa: SLF001
        assert thread is not None and thread.is_alive()
    assert not thread.is_alive()
