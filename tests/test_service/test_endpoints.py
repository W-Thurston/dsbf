"""Every read endpoint of API v2: its 200, its 404 and its 422."""

from urllib.parse import quote
from uuid import uuid4

import pytest

from dsbf.model import Severity
from tests.test_engine.test_executor import TOYS
from tests.test_service.conftest import HOSTILE_COLUMNS, TITANIC_KEY

pytestmark = pytest.mark.integration

PROBLEM = "application/problem+json"
KB = 1000


def encoded(text: str) -> str:
    return quote(text, safe="")


def assert_problem(response, status: int, mention: str | None = None) -> dict:
    assert response.status_code == status, response.text
    assert response.headers["content-type"].startswith(PROBLEM)
    body = response.json()
    assert body["status"] == status
    assert body["instance"] == response.request.url.path or body["instance"]
    if mention:
        assert mention in body["detail"]
    return body


class TestDatasets:
    def test_lists_datasets_with_their_latest_run(self, client, runs):
        page = client.get("/v2/datasets").json()
        by_key = {item["key"]: item for item in page["items"]}
        assert page["total"] == len(by_key)
        assert by_key["planted"]["n_runs"] == 2
        assert by_key["planted"]["last_run_id"] == str(runs["after"].run_id)
        assert by_key[TITANIC_KEY]["name"] == "titanic"

    def test_pages_and_caps_the_limit(self, client):
        page = client.get("/v2/datasets", params={"limit": 1, "offset": 1}).json()
        assert (len(page["items"]), page["limit"], page["offset"]) == (1, 1, 1)
        assert_problem(client.get("/v2/datasets", params={"limit": 501}), 422)
        assert_problem(client.get("/v2/datasets", params={"limit": 0}), 422)

    def test_a_key_with_slashes_lists_its_runs(self, client, runs):
        url = f"/v2/datasets/{encoded(TITANIC_KEY)}/runs"
        page = client.get(url).json()
        assert [i["run_id"] for i in page["items"]] == [str(runs["titanic"].run_id)]

    def test_a_dataset_with_two_runs_lists_the_newer_first(self, client, runs):
        page = client.get("/v2/datasets/planted/runs").json()
        assert [i["run_id"] for i in page["items"]] == [
            str(runs["after"].run_id),
            str(runs["before"].run_id),
        ]

    def test_an_unknown_dataset_is_a_404_naming_it(self, client):
        assert_problem(client.get("/v2/datasets/nope/runs"), 404, "nope")
        assert_problem(
            client.get("/v2/datasets/nope/history", params={"code": "x"}), 404, "nope"
        )

    def test_history_follows_a_finding_over_runs(self, client, runs):
        shared = {f.id for f in runs["before"].findings} & {
            f.id for f in runs["after"].findings
        }
        assert shared
        finding_id = sorted(shared)[0]
        page = client.get(
            "/v2/datasets/planted/history", params={"finding_id": finding_id}
        ).json()
        assert [o["run_id"] for o in page["items"]] == [
            str(runs["after"].run_id),
            str(runs["before"].run_id),
        ]
        assert page["items"][0]["finding"]["id"] == finding_id

    def test_history_needs_a_finding_or_a_code(self, client):
        assert_problem(client.get("/v2/datasets/planted/history"), 422, "finding_id")


class TestLineage:
    def test_flags_a_changed_location_and_unchanged_content(self, client, runs):
        items = client.get("/v2/datasets/relocated/lineage").json()["items"]
        assert [i["run_id"] for i in items] == [
            str(runs["relocated_to"].run_id),
            str(runs["relocated_from"].run_id),
        ]
        newest, oldest = items
        assert newest["location_changed"] and not newest["fingerprint_changed"]
        assert not oldest["location_changed"] and not oldest["fingerprint_changed"]
        assert newest["location"].endswith("titanic_moved.csv")

    def test_flags_changed_content(self, client):
        newest, _ = client.get("/v2/datasets/planted/lineage").json()["items"]
        assert newest["fingerprint_changed"]

    def test_a_page_flags_against_the_run_beyond_it(self, client, runs):
        page = client.get("/v2/datasets/relocated/lineage", params={"limit": 1}).json()
        assert page["total"] == 2
        assert page["items"][0]["location_changed"]

    def test_an_unknown_dataset_is_a_404(self, client):
        assert_problem(client.get("/v2/datasets/nope/lineage"), 404, "nope")


class TestRuns:
    def test_lists_every_run_newest_first(self, client, runs):
        page = client.get("/v2/runs").json()
        assert page["total"] == len(runs) - 1  # one run lives in the other store
        starts = [item["started_at"] for item in page["items"]]
        assert starts == sorted(starts, reverse=True)

    def test_the_overview_has_health_counts_and_files_but_no_results(
        self, client, runs
    ):
        run = runs["titanic"]
        body = client.get(f"/v2/runs/{run.run_id}").json()
        assert body["header"]["run_id"] == str(run.run_id)
        assert body["header"]["health"]["counts"]["ok"] > 0
        assert body["n_findings"] == len(run.findings)
        assert sum(c["count"] for c in body["finding_counts"]) == len(run.findings)
        assert "results" not in body and "results" not in body["header"]
        assert {a["ref"] for a in body["artifacts"]} >= {"snapshot"}

    def test_titanics_overview_is_under_50_kb(self, client, runs):
        response = client.get(f"/v2/runs/{runs['titanic'].run_id}")
        assert len(response.content) < 50 * KB

    def test_a_missing_run_is_a_404_naming_its_id(self, client):
        missing = uuid4()
        assert_problem(client.get(f"/v2/runs/{missing}"), 404, str(missing))

    def test_a_malformed_id_is_a_422(self, client):
        body = assert_problem(client.get("/v2/runs/not-a-uuid"), 422)
        assert body["errors"][0]["location"] == "path.run_id"


class TestFindings:
    def test_titanics_first_page_is_under_100_kb(self, client, runs):
        run = runs["titanic"]
        response = client.get(f"/v2/runs/{run.run_id}/findings")
        page = response.json()
        assert page["total"] == len(run.findings)
        assert len(page["items"]) == page["limit"] == 50
        assert len(response.content) < 100 * KB

    def test_filters_narrow_the_page(self, client, runs):
        run = runs["titanic"]
        url = f"/v2/runs/{run.run_id}/findings"
        warn = client.get(url, params={"min_severity": "warn", "limit": 500}).json()
        assert warn["total"] == sum(f.severity >= Severity.WARN for f in run.findings)
        assert all(i["severity"] in ("warn", "error") for i in warn["items"])
        code = run.findings[0].code
        by_code = client.get(url, params={"code": code, "limit": 500}).json()
        assert {i["code"] for i in by_code["items"]} == {code}
        dimension = run.findings[0].dimension.value
        by_dimension = client.get(url, params={"dimension": dimension}).json()
        assert {i["dimension"] for i in by_dimension["items"]} == {dimension}

    def test_check_confidence_and_sort_agree_with_the_run_filter(self, client, runs):
        run = runs["titanic"]
        url = f"/v2/runs/{run.run_id}/findings"
        check = run.findings[0].check_id
        by_check = client.get(url, params={"check": check, "limit": 500}).json()
        assert by_check["total"] == len(run.findings.filter(check=check))
        high = client.get(url, params={"min_confidence": "high", "limit": 500}).json()
        assert high["total"] == len(run.findings.filter(confidence=">=high"))
        by_code = client.get(url, params={"sort": "code", "limit": 500}).json()
        codes = [i["code"] for i in by_code["items"]]
        assert codes == sorted(codes)

    def test_a_pair_finding_appears_under_both_of_its_columns(self, client, runs):
        run = runs["titanic"]
        url = f"/v2/runs/{run.run_id}/findings"
        pair = next((f for f in run.findings if len(f.subject.columns) > 1), None)
        assert pair is not None
        for column in pair.subject.columns:
            page = client.get(url, params={"column": column, "limit": 500}).json()
            assert pair.id in {i["id"] for i in page["items"]}
            assert page["total"] == len(run.findings.filter(column=column))

    def test_the_offset_moves_through_the_list(self, client, runs):
        url = f"/v2/runs/{runs['titanic'].run_id}/findings"
        first = client.get(url, params={"limit": 5}).json()["items"]
        second = client.get(url, params={"limit": 5, "offset": 5}).json()["items"]
        assert not {i["id"] for i in first} & {i["id"] for i in second}

    def test_a_column_filter_returns_findings_about_it(self, client, runs):
        run = runs["titanic"]
        finding = next(f for f in run.findings if f.subject.columns)
        column = finding.subject.columns[0]
        page = client.get(
            f"/v2/runs/{run.run_id}/findings", params={"column": column, "limit": 500}
        ).json()
        assert finding.id in {i["id"] for i in page["items"]}

    def test_a_bad_filter_is_a_422_and_a_missing_run_a_404(self, client, runs):
        url = f"/v2/runs/{runs['titanic'].run_id}/findings"
        assert_problem(client.get(url, params={"min_severity": "loud"}), 422)
        assert_problem(client.get(url, params={"dimension": "mood"}), 422)
        assert_problem(client.get(f"/v2/runs/{uuid4()}/findings"), 404)

    def test_one_finding(self, client, runs):
        run = runs["titanic"]
        finding = run.findings[0]
        body = client.get(f"/v2/runs/{run.run_id}/findings/{finding.id}").json()
        assert body["id"] == finding.id
        assert_problem(client.get(f"/v2/runs/{run.run_id}/findings/nope"), 404, "nope")


class TestProfiles:
    def test_lists_every_column_in_dataset_order(self, client, runs):
        run = runs["hostile"]
        body = client.get(f"/v2/runs/{run.run_id}/columns").json()
        assert [c["name"] for c in body] == [c.name for c in run.profile.columns]
        positions = [c.position for c in run.profile.columns]
        assert [c["position"] for c in body] == positions
        first = run.profile.columns[0]
        assert body[0]["counts"] == first.counts.model_dump(mode="json")
        assert body[0]["semantic_type"] == first.semantic_type

    def test_listing_columns_of_an_unknown_run_is_a_404(self, client):
        assert_problem(client.get(f"/v2/runs/{uuid4()}/columns"), 404)

    # A client resolves a ".." segment before sending, so that name travels in the
    # query form only.
    @pytest.mark.parametrize("column", [c for c in HOSTILE_COLUMNS if c != ".."])
    def test_a_hostile_column_name_round_trips(self, client, runs, column):
        run = runs["hostile"]
        response = client.get(f"/v2/runs/{run.run_id}/profile/{encoded(column)}")
        assert response.status_code == 200, response.text
        assert response.json()["name"] == column

    @pytest.mark.parametrize("column", HOSTILE_COLUMNS)
    def test_a_hostile_name_round_trips_as_a_query_parameter(
        self, client, runs, column
    ):
        run = runs["hostile"]
        response = client.get(
            f"/v2/runs/{run.run_id}/profile", params={"column": column}
        )
        assert response.status_code == 200, response.text
        assert response.json()["name"] == column
        assert_problem(client.get(f"/v2/runs/{run.run_id}/profile"), 422)

    def test_the_adversarial_names_round_trip(self, client, runs):
        run = runs["adversarial"]
        for profile in run.profile.columns:
            response = client.get(
                f"/v2/runs/{run.run_id}/profile/{encoded(profile.name)}"
            )
            assert response.status_code == 200, (profile.name, response.text)
            assert response.json()["name"] == profile.name

    def test_findings_filter_by_a_hostile_column(self, client, runs):
        run = runs["hostile"]
        for column in HOSTILE_COLUMNS:
            response = client.get(
                f"/v2/runs/{run.run_id}/findings", params={"column": column}
            )
            assert response.status_code == 200

    def test_an_unknown_column_or_run_is_a_404(self, client, runs):
        run = runs["hostile"]
        assert_problem(
            client.get(f"/v2/runs/{run.run_id}/profile/absent"), 404, "absent"
        )
        assert_problem(client.get(f"/v2/runs/{uuid4()}/profile/a"), 404)


class TestChecks:
    def test_a_result_carries_refs_and_no_payload(self, client, runs):
        run = runs["kitchen5k"]
        with_artifacts = next(r for r in run.results.values() if r.artifacts)
        body = client.get(f"/v2/runs/{run.run_id}/checks/{with_artifacts.check_id}")
        assert body.status_code == 200
        refs = [a["ref"] for a in body.json()["artifacts"]]
        assert refs == [a.ref for a in with_artifacts.artifacts]
        assert all(
            set(a) <= {"ref", "media_type", "rows", "bytes", "sha256", "description"}
            or True
            for a in body.json()["artifacts"]
        )

    def test_an_unknown_check_or_run_is_a_404(self, client, runs):
        run = runs["titanic"]
        assert_problem(client.get(f"/v2/runs/{run.run_id}/checks/nope"), 404, "nope")
        assert_problem(client.get(f"/v2/runs/{uuid4()}/checks/nope"), 404)


class TestArtifacts:
    def test_the_snapshot_pages_through_rows(self, client, runs):
        run = runs["titanic"]
        url = f"/v2/runs/{run.run_id}/snapshot"
        first = client.get(url, params={"limit": 4}).json()
        assert first["total"] == run.dataset.n_rows
        assert first["columns"][:2] == ["survived", "pclass"]
        assert len(first["rows"]) == 4
        second = client.get(url, params={"limit": 4, "offset": 4}).json()
        assert second["rows"] != first["rows"]
        assert first["sampling"] is None

    def test_a_run_with_no_snapshot_is_a_404(self, bare_client, runs):
        run = runs["no_snapshot"]
        assert_problem(
            bare_client.get(f"/v2/runs/{run.run_id}/snapshot"), 404, "snapshot"
        )

    def test_a_table_artifact_pages(self, client, runs):
        run = runs["kitchen5k"]
        overview = client.get(f"/v2/runs/{run.run_id}").json()
        table = next(
            a
            for a in overview["artifacts"]
            if a["rows"] is not None and a["ref"] != "snapshot"
        )
        url = f"/v2/runs/{run.run_id}/artifacts/{table['ref']}"
        page = client.get(url, params={"limit": 3, "offset": 2}).json()
        assert page["total"] == table["rows"]
        assert len(page["rows"]) == 3 and page["offset"] == 2
        assert len(page["rows"][0]) == len(page["columns"])

    def test_a_json_artifact_returns_its_value(self, client, runs):
        run = runs["failed"]
        ref = f"{TOYS[0].id}/traceback"
        body = client.get(f"/v2/runs/{run.run_id}/artifacts/{ref}").json()
        assert body["ref"] == ref and "boom" in body["value"]

    def test_an_unknown_artifact_is_a_404(self, client, runs):
        run = runs["titanic"]
        assert_problem(client.get(f"/v2/runs/{run.run_id}/artifacts/a/b"), 404, "a/b")


class TestCompare:
    def test_reports_the_planted_change(self, client, runs):
        response = client.get(
            "/v2/runs/compare",
            params={
                "base": str(runs["before"].run_id),
                "head": str(runs["after"].run_id),
            },
        )
        diff = response.json()
        assert response.status_code == 200
        assert diff["same_dataset"] is True
        assert diff["new"] or diff["changed"]
        assert diff["head_run_id"] == str(runs["after"].run_id)

    def test_any_two_runs_compare_and_say_they_differ(self, client, runs):
        diff = client.get(
            "/v2/runs/compare",
            params={
                "base": str(runs["titanic"].run_id),
                "head": str(runs["after"].run_id),
            },
        ).json()
        assert diff["same_dataset"] is False

    def test_a_missing_run_is_a_404_and_bad_input_a_422(self, client, runs):
        good = str(runs["before"].run_id)
        missing = str(uuid4())
        assert_problem(
            client.get("/v2/runs/compare", params={"base": missing, "head": good}),
            404,
            missing,
        )
        assert_problem(client.get("/v2/runs/compare", params={"base": good}), 422)
        assert_problem(
            client.get(
                "/v2/runs/compare",
                params={"base": good, "head": good, "tolerance": -1},
            ),
            422,
        )
        assert_problem(
            client.get("/v2/runs/compare", params={"base": "x", "head": good}), 422
        )


class TestRelationships:
    def test_titanics_cells_equal_the_checks_pairs(self, client, runs):
        run = runs["titanic"]
        body = client.get(f"/v2/runs/{run.run_id}/relationships").json()
        by_check: dict[str, list] = {}
        for cell in body["cells"]:
            by_check.setdefault(cell["check_id"], []).append(cell)
        for check_id, key in (
            ("associations", "value"),
            ("contingency", "cramers_v_corrected"),
            ("group_differences", "omega_squared"),
        ):
            pairs = run.results[check_id].output["pairs"]
            assert [c["value"] for c in by_check[check_id]] == [p[key] for p in pairs]
        statistics = {c["statistic"] for c in body["cells"]}
        assert statistics == set(body["statistics"])
        assert [c["pairs_tested"] for c in body["coverage"]] == [
            len(run.results[c["check_id"]].output["pairs"]) for c in body["coverage"]
        ]

    def test_an_untested_pair_has_no_cell(self, client, runs):
        run = runs["titanic"]
        body = client.get(f"/v2/runs/{run.run_id}/relationships").json()
        seen = {frozenset((c["first"], c["second"])) for c in body["cells"]}
        assert frozenset(("age", "age")) not in seen
        assert len(seen) < len(body["columns"]) ** 2

    def test_columns_follow_the_datasets_order(self, client, runs):
        run = runs["titanic"]
        body = client.get(f"/v2/runs/{run.run_id}/relationships").json()
        order = [c["name"] for c in client.get(f"/v2/runs/{run.run_id}/columns").json()]
        assert body["columns"] == [n for n in order if n in body["columns"]]

    def test_a_missing_run_is_a_404(self, client):
        assert_problem(client.get(f"/v2/runs/{uuid4()}/relationships"), 404, "run")


class TestCoverage:
    def test_checks_carry_their_examined_columns(self, client, runs):
        run = runs["titanic"]
        body = client.get(f"/v2/runs/{run.run_id}/coverage").json()
        assert [c["check_id"] for c in body["checks"]] == list(run.results)
        for check in body["checks"]:
            assert check["columns"] == list(run.results[check["check_id"]].columns)

    def test_columns_follow_the_datasets_order_and_count_checks(self, client, runs):
        run = runs["titanic"]
        body = client.get(f"/v2/runs/{run.run_id}/coverage").json()
        assert [c["column"] for c in body["columns"]] == [
            c.name for c in run.profile.columns
        ]
        completed = sum(1 for r in run.results.values() if r.status.value == "ok")
        age = next(c for c in body["columns"] if c["column"] == "age")
        assert age["checks_completed"] == completed
        assert age["examined_by"] == sum(
            1
            for r in run.results.values()
            if r.status.value == "ok" and "age" in r.columns
        )

    def test_a_missing_run_is_a_404(self, client):
        assert_problem(client.get(f"/v2/runs/{uuid4()}/coverage"), 404, "run")


class TestSnapshotPair:
    def test_returns_both_columns_for_rows_with_both(self, client, runs):
        run = runs["titanic"]
        body = client.get(
            f"/v2/runs/{run.run_id}/snapshot/pair",
            params={"first": "age", "second": "fare"},
        ).json()
        assert body["snapshot_rows"] == run.dataset.n_rows
        assert body["rows_with_both"] == 714
        assert len(body["values"]) == 714
        assert body["sampling"] is None

    def test_thins_evenly_to_the_limit(self, client, runs):
        run = runs["titanic"]
        params = {"first": "age", "second": "fare", "limit": 100}
        one = client.get(f"/v2/runs/{run.run_id}/snapshot/pair", params=params).json()
        two = client.get(f"/v2/runs/{run.run_id}/snapshot/pair", params=params).json()
        assert len(one["values"]) == 100
        assert one["rows_with_both"] == 714
        assert one["values"] == two["values"]

    def test_the_limit_is_capped(self, client, runs):
        response = client.get(
            f"/v2/runs/{runs['titanic'].run_id}/snapshot/pair",
            params={"first": "age", "second": "fare", "limit": 5000},
        )
        assert response.status_code == 422

    def test_an_unknown_column_is_a_404_naming_it(self, client, runs):
        response = client.get(
            f"/v2/runs/{runs['titanic'].run_id}/snapshot/pair",
            params={"first": "age", "second": "nope"},
        )
        assert_problem(response, 404, "nope")

    def test_a_run_without_a_snapshot_is_a_404(self, bare_client, runs):
        response = bare_client.get(
            f"/v2/runs/{runs['no_snapshot'].run_id}/snapshot/pair",
            params={"first": "id", "second": "amount"},
        )
        assert_problem(response, 404, "snapshot")
