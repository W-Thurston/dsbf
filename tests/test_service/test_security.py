"""A service with no authentication answers only local requests."""

import pytest
from fastapi.testclient import TestClient

from dsbf.service import create_app
from tests.test_service.test_endpoints import PROBLEM

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "127.0.0.1:8765", "localhost", "LOCALHOST:80", "[::1]:9"]
)
def test_a_local_host_is_served(store, host):
    client = TestClient(create_app(store[0]), base_url="http://127.0.0.1")
    assert client.get("/v2/datasets", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize(
    "host", ["evil.example", "evil.example:8765", "127.0.0.1.evil.example", ""]
)
def test_another_host_is_refused_with_a_problem(store, host):
    client = TestClient(create_app(store[0]), base_url="http://127.0.0.1")
    response = client.get("/v2/datasets", headers={"Host": host})
    assert response.status_code == 421
    assert response.headers["content-type"] == PROBLEM
    assert response.json()["status"] == 421


def test_hosts_can_be_widened_by_the_caller(store):
    app = create_app(store[0], allowed_hosts=("dsbf.internal",))
    client = TestClient(app, base_url="http://dsbf.internal")
    assert client.get("/v2/datasets").status_code == 200
    assert (
        TestClient(app, base_url="http://127.0.0.1").get("/v2/datasets").status_code
        == 421
    )


def test_there_is_no_cors_header_by_default(client):
    response = client.get("/v2/datasets", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_a_configured_origin_gets_cors_headers_and_others_do_not(store):
    app = create_app(store[0], cors_origins=("http://localhost:5173",))
    client = TestClient(app, base_url="http://127.0.0.1")
    allowed = client.get("/v2/datasets", headers={"Origin": "http://localhost:5173"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.get("/v2/datasets", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_the_service_offers_reads_only(client):
    for method in ("post", "put", "delete", "patch"):
        response = getattr(client, method)("/v2/runs")
        assert response.status_code == 405
        assert response.headers["content-type"].startswith(PROBLEM)


def test_an_unknown_path_is_a_problem_too(client):
    response = client.get("/v2/nothing")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith(PROBLEM)


def test_the_built_dashboard_is_served_after_the_api(store, tmp_path):
    (tmp_path / "index.html").write_text("<title>dashboard</title>")
    app = create_app(store[0], static_dir=tmp_path)
    client = TestClient(app, base_url="http://127.0.0.1")
    assert "dashboard" in client.get("/").text
    assert client.get("/v2/datasets").status_code == 200


def test_a_page_route_gets_the_dashboard_and_a_missing_file_does_not(store, tmp_path):
    (tmp_path / "index.html").write_text("<title>dashboard</title>")
    client = TestClient(
        create_app(store[0], static_dir=tmp_path), base_url="http://127.0.0.1"
    )
    assert "dashboard" in client.get("/runs/0b3a5e4c").text
    assert client.get("/assets/missing.js").status_code == 404
    assert client.get("/v2/runs/not-a-run").status_code == 422
