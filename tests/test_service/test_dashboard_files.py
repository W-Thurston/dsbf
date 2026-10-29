"""The service serves the built dashboard, or says the dashboard is missing."""

import pytest
from fastapi.testclient import TestClient

from dsbf.service import create_app
from dsbf.service import serve as serve_module

pytestmark = pytest.mark.integration


@pytest.fixture
def built(tmp_path):
    (tmp_path / "index.html").write_text('<div id="app"></div>')
    (tmp_path / "app.js").write_text("console.log(1)")
    return tmp_path


def test_a_page_route_gets_the_dashboard_page(store, built):
    client = TestClient(
        create_app(store[0], static_dir=built), base_url="http://127.0.0.1"
    )

    assert client.get("/").text == '<div id="app"></div>'
    assert client.get("/runs/abc").text == '<div id="app"></div>'
    assert client.get("/app.js").text == "console.log(1)"
    assert client.get("/missing.js").status_code == 404


def test_without_the_dashboard_the_root_says_so_and_the_api_still_answers(store):
    client = TestClient(create_app(store[0]), base_url="http://127.0.0.1")

    page = client.get("/")
    assert page.status_code == 200
    assert "dashboard is not installed" in page.text
    assert client.get("/v2/datasets").status_code == 200
    assert "/" not in client.get("/v2/openapi.json").json()["paths"]


def test_the_packaged_dashboard_serves_a_wheel_with_no_checkout_build(
    tmp_path, monkeypatch
):
    packaged = tmp_path / "service" / "dashboard"
    packaged.mkdir(parents=True)
    (packaged / "index.html").write_text("x")
    monkeypatch.setattr(
        serve_module, "__file__", str(tmp_path / "service" / "serve.py")
    )

    assert serve_module.dashboard_directory() == packaged


def test_a_checkout_build_wins_over_the_packaged_one(tmp_path, monkeypatch):
    service = tmp_path / "src" / "dsbf" / "service"
    (service / "dashboard").mkdir(parents=True)
    (service / "dashboard" / "index.html").write_text("packaged")
    dist = tmp_path / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("built")
    monkeypatch.setattr(serve_module, "__file__", str(service / "serve.py"))

    assert serve_module.dashboard_directory() == dist


def test_no_index_means_no_dashboard(tmp_path, monkeypatch):
    (tmp_path / "dashboard").mkdir()
    monkeypatch.setattr(serve_module, "__file__", str(tmp_path / "serve.py"))

    assert serve_module.dashboard_directory() is None
