"""A check attaches bulky data; it rides on the run, the cache and the files."""

import os

import pyarrow as pa
import pytest

from dsbf.engine.cache import DiskCache
from dsbf.model import (
    CheckResult,
    CheckStatus,
    Confidence,
    Evidence,
    Run,
    Severity,
    Subject,
)
from tests.test_engine import process_toys as toys
from tests.test_engine.test_cache import go
from tests.test_engine.toys import declare, run_plan

TABLE = pa.table({"n": [1, 2, 3], "square": [1, 4, 9]})


def attaching(*payloads, details_ref=None):
    """A check that attaches ``(name, payload)`` pairs and reports a finding."""

    def function(ctx, params):
        refs = [
            ctx.attach(name, payload, description=f"the {name}")
            for name, payload in payloads
        ]
        if details_ref is not None:
            ctx.finding(
                code="attached",
                subject=Subject.dataset(),
                severity=Severity.INFO,
                confidence=Confidence.HIGH,
                title="attached",
                message="attached",
                evidence=Evidence(
                    metrics={},
                    n_used=1,
                    n_total=1,
                    method="attach",
                    details_ref=refs[details_ref],
                ),
            )
        return None

    return declare("attaches", function=function)


@pytest.mark.unit
class TestAttach:
    def test_a_table_and_a_value_get_a_ref_with_their_media_type_rows_and_bytes(self):
        run = run_plan([attaching(("table", TABLE), ("value", {"a": [1, None]}))])

        table, value = run.results["attaches"].artifacts
        assert table.ref == "attaches/table"
        assert table.media_type == "application/vnd.apache.parquet"
        assert table.rows == 3 and table.bytes > 0
        assert table.description == "the table"
        assert value.media_type == "application/json" and value.rows is None
        assert run.artifact("attaches/table").equals(TABLE)
        assert run.artifact("attaches/value") == {"a": [1, None]}

    def test_the_run_json_holds_the_refs_and_not_the_payload(self):
        run = run_plan([attaching(("table", TABLE))])

        text = run.model_dump_json()

        assert '"attaches/table"' in text and '"square"' not in text

    def test_a_non_finite_number_in_a_value_becomes_null(self):
        run = run_plan([attaching(("value", [1.0, float("nan")]))])

        assert run.artifact("attaches/value") == [1.0, None]

    @pytest.mark.parametrize("name", ["", "Upper", "a/b", "a b", ".x", "x."])
    def test_a_bad_name_is_a_contract_error(self, name):
        run = run_plan([attaching((name, TABLE))])

        assert run.results["attaches"].status is CheckStatus.ERROR
        assert "artifact name" in run.results["attaches"].notes[0].message

    def test_a_name_used_twice_is_a_contract_error(self):
        run = run_plan([attaching(("t", TABLE), ("t", TABLE))])

        assert run.results["attaches"].status is CheckStatus.ERROR

    def test_a_payload_of_another_type_is_a_contract_error(self):
        run = run_plan([attaching(("t", object()))])

        assert run.results["attaches"].status is CheckStatus.ERROR

    def test_a_check_that_stops_keeps_no_artifacts_of_its_own(self):
        def function(ctx, params):
            ctx.attach("t", TABLE)
            return ctx.stop(CheckStatus.SKIPPED, "nope", "not today")

        run = run_plan([declare("stops", function=function)])

        assert run.results["stops"].status is CheckStatus.SKIPPED
        assert run.results["stops"].artifacts == ()


@pytest.mark.unit
class TestRunContract:
    def test_a_details_ref_names_an_artifact_of_the_run(self):
        run = run_plan([attaching(("t", TABLE), details_ref=0)])

        (finding,) = run.findings
        assert finding.evidence.details_ref == "attaches/t"

    def test_a_dangling_details_ref_is_rejected(self):
        run = run_plan([attaching(("t", TABLE), details_ref=0)])
        data = run.model_dump(mode="json")
        data["results"]["attaches"]["artifacts"] = []

        with pytest.raises(ValueError, match="points at artifact"):
            Run.model_validate(data)

    def test_an_unknown_ref_lists_the_refs_the_run_has(self):
        run = run_plan([attaching(("t", TABLE))])

        with pytest.raises(KeyError, match=r"attaches/t"):
            run.artifact("attaches/other")

    def test_a_result_cannot_carry_another_checks_artifact(self):
        run = run_plan([attaching(("t", TABLE))])
        data = run.results["attaches"].model_dump(mode="json")
        data["artifacts"][0]["ref"] = "other/t"

        with pytest.raises(ValueError, match="does not belong"):
            CheckResult.model_validate(data)


@pytest.mark.unit
class TestFiles:
    def test_a_written_run_reads_back_with_equal_artifacts(self, tmp_path):
        run = run_plan([attaching(("table", TABLE), ("value", {"k": [1, 2]}))])

        run.write(tmp_path)
        back = Run.read(tmp_path)

        assert back.results["attaches"].artifacts == run.results["attaches"].artifacts
        assert (tmp_path / "artifacts/attaches/table.parquet").is_file()
        assert (tmp_path / "artifacts/attaches/value.json").is_file()
        assert back.artifact("attaches/table").equals(TABLE)
        assert back.artifact("attaches/value") == {"k": [1, 2]}

    def test_a_missing_file_says_which_artifact(self, tmp_path):
        run_plan([attaching(("table", TABLE))]).write(tmp_path)
        (tmp_path / "artifacts/attaches/table.parquet").unlink()

        with pytest.raises(KeyError, match="attaches/table"):
            Run.read(tmp_path).artifact("attaches/table")

    def test_a_read_run_can_be_written_again_beside_itself(self, tmp_path):
        run_plan([attaching(("table", TABLE))]).write(tmp_path)

        Run.read(tmp_path).write(tmp_path)

        assert Run.read(tmp_path).artifact("attaches/table").equals(TABLE)

    def test_a_run_without_artifacts_writes_no_folder(self, tmp_path):
        run_plan([declare("plain")]).write(tmp_path)

        assert not (tmp_path / "artifacts").exists()
        assert Run.read(tmp_path).artifacts == ()


@pytest.mark.unit
class TestCache:
    def test_a_hit_returns_the_artifacts_intact(self, tmp_path):
        checks = [attaching(("table", TABLE), ("value", {"a": 1}))]
        first = go(checks, DiskCache(tmp_path))

        again = go(checks, DiskCache(tmp_path))

        for ref in ("attaches/table", "attaches/value"):
            hit, miss = again.artifact(ref), first.artifact(ref)
            assert hit.equals(miss) if isinstance(hit, pa.Table) else hit == miss
        assert (
            again.results["attaches"].artifacts == first.results["attaches"].artifacts
        )

    def test_a_hit_does_not_call_the_check(self, tmp_path):
        calls = []

        def function(ctx, params):
            calls.append(1)
            ctx.attach("t", TABLE)

        checks = [declare("attaches", function=function)]
        go(checks, DiskCache(tmp_path))
        go(checks, DiskCache(tmp_path))

        assert calls == [1]

    def test_a_missing_payload_file_is_a_miss_and_the_check_runs_again(self, tmp_path):
        calls = []

        def function(ctx, params):
            calls.append(1)
            ctx.attach("t", TABLE)

        checks = [declare("attaches", function=function)]
        go(checks, DiskCache(tmp_path))
        for path in tmp_path.rglob("*.t.parquet"):
            path.unlink()

        run = go(checks, DiskCache(tmp_path))

        assert calls == [1, 1]
        assert run.artifact("attaches/t").equals(TABLE)


@pytest.mark.integration
class TestProcessWorker:
    def test_a_process_isolated_check_returns_its_artifacts(self):
        run = run_plan([toys.py_attach], workers=1)

        result = run.results["py_attach"]
        assert result.status is CheckStatus.OK
        assert run.artifact("py_attach/squares").equals(
            pa.table({"n": [1, 2, 3], "square": [1, 4, 9]})
        )
        assert run.artifact("py_attach/meta")["pid"] != os.getpid()
        assert run.findings[0].evidence.details_ref == "py_attach/squares"
