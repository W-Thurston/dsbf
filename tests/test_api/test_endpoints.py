# tests/test_api/test_endpoints.py

"""Exercise every read endpoint of the FastAPI app against a seeded database.

The database is built with ``persist_run_from_dict`` from a few hand-written task
results, so the test needs no engine run. It covers the happy path of each endpoint
and the 404 and 400 responses, so a framework upgrade that changes routing,
validation or file serving shows up here.
"""

import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dsbf.api import db, main
from dsbf.storage.writer import persist_run_from_dict

pytestmark = pytest.mark.integration

# A 1x1 PNG, enough for the figure endpoint to serve a real file.
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cfc000000301010018dd8db00000000049454e44ae426082"
)

DATASET = "toy"
RUN_A = "20260101_000000"
RUN_B = "20260102_000000"


def _task_results(figure_path: str, score: float) -> dict:
    return {
        "data_quality_scorer": {
            "status": "success",
            "summary": {"score": score},
            "data": {
                "total_columns": 3,
                "categories": {
                    "completeness": {
                        "level": "amber",
                        "affected_count": 1,
                        "pct_affected": 0.33,
                    },
                    "validity": {"level": "green"},
                },
            },
            "guidance": [],
        },
        "ml_readiness_scorer": {
            "status": "success",
            "summary": {},
            "data": {"overall_score": 80, "readiness_gate": "needs_work"},
            "guidance": [],
        },
        "compute_pairwise_associations": {
            "status": "success",
            "summary": {"pairs": 2},
            "data": {
                "a|b": {
                    "metric": 0.9,
                    "metric_type": "pearson_r",
                    "strength": "strong",
                },
                "a|c": {
                    "metric": 0.1,
                    "metric_type": "pearson_r",
                    "strength": "negligible",
                },
                "__correlation_matrix__": {"a": {"b": 0.9}},
            },
            "guidance": [],
        },
        "generate_dataset_summary_plots": {
            "status": "success",
            "summary": {},
            "data": {"missingness": {"png": figure_path}},
            "guidance": [],
        },
    }


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    db_path = tmp_path / "api.db"
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text("a,b,c\n1,2,x\n3,4,y\n5,6,z\n")
    figure_path = tmp_path / "missingness.png"
    figure_path.write_bytes(PNG_BYTES)

    for run_key, score in ((RUN_A, 0.5), (RUN_B, 0.75)):
        persist_run_from_dict(
            run_key=run_key,
            dataset_name=DATASET,
            source_path=str(csv_path),
            run_meta={
                "profiling_depth": "full",
                "inferred_stage": "raw",
                "row_count": 3,
                "col_count": 3,
                "quality_score": score,
                "ran_at": f"{run_key[:4]}-{run_key[4:6]}-{run_key[6:8]}T00:00:00",
            },
            results=_task_results(str(figure_path), score),
            db_path=db_path,
        )

    # A few handlers call the db helpers without the module override, so point both
    # the override and the environment variable at the seeded database.
    monkeypatch.setattr(main, "_DB_PATH", str(db_path))
    monkeypatch.setenv("DSBF_DB_PATH", str(db_path))
    return TestClient(main.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_datasets_and_runs(client):
    datasets = client.get("/api/datasets").json()
    assert [d["name"] for d in datasets] == [DATASET]
    assert datasets[0]["run_count"] == 2
    assert datasets[0]["latest_run_key"] == RUN_B

    assert client.get(f"/api/datasets/{DATASET}").status_code == 200
    runs = client.get(f"/api/datasets/{DATASET}/runs").json()
    assert {r["run_key"] for r in runs} == {RUN_A, RUN_B}

    run = client.get(f"/api/runs/{RUN_A}").json()
    assert run["row_count"] == 3
    assert run["col_count"] == 3


def test_task_results(client):
    tasks = client.get(f"/api/runs/{RUN_A}/tasks").json()
    assert set(tasks) == {
        "data_quality_scorer",
        "ml_readiness_scorer",
        "compute_pairwise_associations",
        "generate_dataset_summary_plots",
    }
    task = client.get(f"/api/runs/{RUN_A}/tasks/data_quality_scorer").json()
    assert task["status"] == "success"
    assert task["data"]["total_columns"] == 3


def test_compare_runs(client):
    response = client.get(
        "/api/runs/compare",
        params={
            "run_keys": [RUN_A, RUN_B],
            "task_name": "data_quality_scorer",
        },
    )
    assert response.status_code == 200
    assert (
        client.get(
            "/api/runs/compare",
            params={"run_keys": [RUN_A], "task_name": "data_quality_scorer"},
        ).status_code
        == 400
    )


def test_dq_status_and_ml_readiness(client):
    dq = client.get(f"/api/runs/{RUN_A}/dq-status").json()
    assert dq["available"] is True
    assert dq["total_columns"] == 3
    assert dq["categories"]["completeness"]["level"] == "amber"
    assert dq["categories"]["validity"]["level"] == "green"

    ml = client.get(f"/api/runs/{RUN_A}/ml-readiness").json()
    assert ml["available"] is True
    assert ml["readiness_gate"] == "needs_work"


def test_associations_and_correlations(client):
    associations = client.get(f"/api/runs/{RUN_A}/associations").json()
    assert associations["unavailable"] is False
    assert set(associations["pairs"]) == {"a|b", "a|c"}

    column = client.get(f"/api/runs/{RUN_A}/associations/a").json()
    assert [row["column"] for row in column["associations"]] == ["b", "c"]
    strong = client.get(
        f"/api/runs/{RUN_A}/associations/a", params={"min_strength": "strong"}
    ).json()
    assert [row["column"] for row in strong["associations"]] == ["b"]

    correlations = client.get(f"/api/runs/{RUN_A}/correlations/a").json()
    assert correlations["unavailable"] is False
    assert correlations["correlations"][0]["column"] == "b"


def test_source_data_endpoints(client):
    sample = client.get(f"/api/runs/{RUN_A}/sample", params={"n": 2}).json()
    assert sample["columns"] == ["a", "b", "c"]
    assert len(sample["rows"]) == 2

    data = client.get(
        f"/api/runs/{RUN_A}/column-data", params={"cols": ["a", "b"]}
    ).json()
    assert data["total_rows"] == 3
    assert client.get(f"/api/runs/{RUN_A}/column-data").status_code == 400


SECRET = "/srv/private/toy.csv: secret failure detail"


def _raise_secret(*args, **kwargs):
    raise RuntimeError(SECRET)


@pytest.mark.parametrize(
    ("path", "params", "key"),
    [
        (f"/api/runs/{RUN_A}/sample", {"n": 2}, "error"),
        (f"/api/runs/{RUN_A}/column-data", {"cols": ["a", "b"]}, "error"),
    ],
)
def test_source_read_failure_hides_exception_text(
    client, monkeypatch, caplog, path, params, key
):
    monkeypatch.setattr(db.pd, "read_csv", _raise_secret)
    with caplog.at_level(logging.ERROR, logger="dsbf.api.db"):
        response = client.get(path, params=params)

    assert response.json()[key] == db.SOURCE_READ_ERROR
    assert "secret" not in response.text
    assert SECRET in caplog.text


def test_correlation_parse_failure_hides_exception_text(client, monkeypatch, caplog):
    monkeypatch.setattr(main.db, "get_task", lambda *a, **k: {"data": "not json"})
    monkeypatch.setattr(main, "_json", SimpleNamespace(loads=_raise_secret))
    with caplog.at_level(logging.ERROR, logger="dsbf.api.main"):
        body = client.get(f"/api/runs/{RUN_A}/correlations/a").json()

    assert body["unavailable"] is True
    assert body["reason"].startswith("Could not parse correlation data")
    assert "secret" not in body["reason"]
    assert SECRET in caplog.text


def test_figures(client):
    figures = client.get(f"/api/runs/{RUN_A}/figures").json()
    assert len(figures) == 1
    assert figures[0]["plot_type"] == "missingness"

    response = client.get(f"/api/figures/{figures[0]['id']}/file")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == PNG_BYTES


@pytest.mark.parametrize(
    "path",
    [
        "/api/datasets/missing",
        "/api/datasets/missing/runs",
        "/api/runs/missing",
        "/api/runs/missing/tasks",
        f"/api/runs/{RUN_A}/tasks/missing",
        "/api/runs/missing/dq-status",
        "/api/runs/missing/ml-readiness",
        "/api/runs/missing/associations",
        "/api/runs/missing/figures",
        "/api/runs/missing/correlations/a",
        "/api/figures/9999/file",
    ],
)
def test_missing_resources_return_404(client, path):
    assert client.get(path).status_code == 404
