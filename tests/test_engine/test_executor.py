"""Executing a plan always completes, and every check ends in an explicit status."""

import dataclasses

import polars as pl
import pytest

from dsbf.data import load
from dsbf.engine import (
    CheckContext,
    Expected,
    Requires,
    SemanticTypes,
    execute,
    run_check,
)
from dsbf.engine import extras as extras_module
from dsbf.model import CheckResult, CheckStatus, Run, Timing
from dsbf.profile import profile_dataset
from tests.test_engine.toys import (
    declare,
    make_dataset,
    null_share,
    numeric_spread,
    plan_for,
    quality_summary,
    run_plan,
    run_toys,
)

pytestmark = pytest.mark.unit

TOYS = (null_share, numeric_spread, quality_summary)


def explode(ctx, params):
    raise RuntimeError("boom")


def broken(toy):
    return dataclasses.replace(toy, function=explode)


def statuses(run: Run) -> dict[str, CheckStatus]:
    return {check_id: result.status for check_id, result in run.results.items()}


class TestCompletion:
    def test_the_toys_run_to_the_results_a_hand_run_gives(self):
        run = run_plan(TOYS)

        by_hand = run_toys()
        assert set(run.results) == set(by_hand)
        for check_id, result in run.results.items():
            expected = by_hand[check_id]
            assert result.status is expected.status is CheckStatus.OK
            assert result.findings == expected.findings
            assert result.output == expected.output
            assert result.params == expected.params
        assert run.health.counts.ok == 3

    def test_the_run_records_its_inputs(self):
        dataset = make_dataset(rows=60, strategy="random", seed=3)

        run = run_plan(TOYS, dataset)

        assert run.dataset.fingerprint == dataset.fingerprint
        assert run.dataset.n_rows == dataset.source_rows
        assert run.dataset.sampling == dataset.sampling
        assert run.plan.dsbf_version == "0.0.0"
        assert run.plan.depth == "standard"
        planned = {c.check_id: c for c in run.plan.checks}
        assert planned["quality_summary"].depends_on == ("null_share", "numeric_spread")
        assert planned["numeric_spread"].params == {"min_std": 0.5}
        assert run.finished_at >= run.started_at

    def test_an_empty_plan_is_an_empty_run(self):
        run = run_plan([])

        assert run.results == {}
        assert run.health.counts.total == 0

    def test_a_profile_of_other_data_is_rejected(self):
        dataset = make_dataset()
        other = profile_dataset(make_dataset(pl.DataFrame({"x": range(30)})))
        plan = plan_for(TOYS, None)

        with pytest.raises(ValueError, match="different columns"):
            execute(plan, dataset, other, dsbf_version="0.0.0")


class TestIsolation:
    @pytest.mark.parametrize("victim", TOYS, ids=lambda toy: toy.id)
    def test_a_forced_exception_still_yields_a_complete_run(self, victim):
        checks = [broken(t) if t.id == victim.id else t for t in TOYS]

        run = run_plan(checks)

        assert set(run.results) == {t.id for t in TOYS}
        result = run.results[victim.id]
        assert result.status is CheckStatus.ERROR
        assert result.findings == () and result.output is None
        (note,) = result.notes
        assert note.code == "check_failed"
        assert note.exception_type == "RuntimeError"
        assert note.message == "Raised RuntimeError: boom"
        assert note.hint
        assert note.detail and "Traceback" in note.detail and "boom" in note.detail
        assert result.params == victim.resolve_params().model_dump(mode="json")
        assert result.timing.started_at is not None
        assert run.health.counts.error == 1

    def test_the_health_summary_lists_the_reason(self):
        run = run_plan([broken(null_share), numeric_spread, quality_summary])

        text = run.render_health()

        assert "error 1" in text and "skipped 1" in text
        assert "null_share" in text and "Raised RuntimeError: boom" in text
        assert "Requires 'null_share' (error)" in text

    def test_an_exception_message_that_is_empty_still_says_what_happened(self):
        def silent(ctx, params):
            raise KeyError

        run = run_plan([declare("silent", function=silent)])

        assert "KeyError" in run.results["silent"].notes[0].message

    def test_a_contract_violation_is_reported_as_a_bug_in_the_check(self):
        def overreach(ctx, params):
            ctx.values("amount")

        run = run_plan([declare("overreach", function=overreach)])

        result = run.results["overreach"]
        assert result.status is CheckStatus.ERROR
        assert result.notes[0].exception_type == "CheckContractError"
        assert "bug in the check" in (result.notes[0].hint or "")

    def test_an_exception_after_findings_drops_them(self):
        def half_done(ctx, params):
            ctx.finding(**_finding_fields())
            raise RuntimeError("late")

        run = run_plan([declare("half_done", function=half_done)])

        assert run.results["half_done"].findings == ()
        assert run.health.counts.error == 1

    def test_a_keyboard_interrupt_is_not_swallowed(self):
        def interrupted(ctx, params):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run_plan([declare("interrupted", function=interrupted)])


def _finding_fields():
    from dsbf.model import Confidence, Evidence, Severity, Subject

    return {
        "code": "x",
        "subject": Subject.dataset(),
        "severity": Severity.INFO,
        "confidence": Confidence.HIGH,
        "title": "t",
        "message": "m",
        "evidence": Evidence(metrics={}, n_used=1, n_total=1, method="m"),
    }


class TestDependents:
    def test_a_hard_dependent_of_an_errored_check_is_skipped_naming_it(self):
        run = run_plan([broken(null_share), numeric_spread, quality_summary])

        summary = run.results["quality_summary"]
        assert summary.status is CheckStatus.SKIPPED
        (note,) = summary.notes
        assert note.code == "dependency_failed"
        assert note.blocked_by == ("null_share",)
        assert run.results["numeric_spread"].status is CheckStatus.OK

    def test_a_soft_upstream_that_errors_leaves_the_dependent_running(self):
        run = run_plan([null_share, broken(numeric_spread), quality_summary])

        assert statuses(run) == {
            "null_share": CheckStatus.OK,
            "numeric_spread": CheckStatus.ERROR,
            "quality_summary": CheckStatus.OK,
        }
        # It reads the absent soft upstream as None, so it counted only null_share.
        assert run.results["quality_summary"].output == {
            "findings_seen": len(run.results["null_share"].findings)
        }

    def test_skipping_spreads_down_the_chain(self):
        chain = [
            broken(declare("first")),
            declare("second", requires=Requires(checks=["first"])),
            declare("third", requires=Requires(checks=["second"])),
        ]

        run = run_plan(chain)

        assert statuses(run) == {
            "first": CheckStatus.ERROR,
            "second": CheckStatus.SKIPPED,
            "third": CheckStatus.SKIPPED,
        }
        assert run.results["third"].notes[0].blocked_by == ("second",)

    def test_a_not_applicable_requirement_leaves_the_dependent_running(self):
        frame = pl.DataFrame({"colour": ["red", "green"] * 15})
        checks = [
            declare("spread", applies_to=SemanticTypes("numeric")),
            declare("rollup", requires=Requires(checks=["spread"])),
        ]

        run = run_plan(checks, make_dataset(frame))

        assert statuses(run) == {
            "spread": CheckStatus.NOT_APPLICABLE,
            "rollup": CheckStatus.OK,
        }

    def test_a_dependent_reads_its_upstream_result(self):
        run = run_plan(TOYS)

        assert run.results["quality_summary"].output == {
            "findings_seen": len(run.results["null_share"].findings)
            + len(run.results["numeric_spread"].findings)
        }


class TestUnavailable:
    @pytest.fixture
    def without_statsmodels(self, monkeypatch):
        real = extras_module.find_spec
        monkeypatch.setattr(
            extras_module,
            "find_spec",
            lambda m: None if m == "statsmodels" else real(m),
        )

    def test_a_missing_extra_is_unavailable_and_the_check_is_not_called(
        self, without_statsmodels
    ):
        called = []

        def acf(ctx, params):
            called.append(True)

        checks = [
            declare("acf", extras=["timeseries"], function=acf),
            declare("after_acf", requires=Requires(checks=["acf"])),
        ]

        run = run_plan(checks)

        assert not called
        assert statuses(run) == {
            "acf": CheckStatus.UNAVAILABLE,
            "after_acf": CheckStatus.SKIPPED,
        }
        note = run.results["acf"].notes[0]
        assert note.code == "extra_missing"
        assert "pip install 'dsbf[timeseries]'" in (note.hint or "")
        assert "Install it with" in run.render_health()


class TestReturnedStatus:
    """A result that says it failed is never counted as ok (the legacy engine did)."""

    def test_a_returned_error_result_counts_as_an_error_and_blocks_dependents(
        self, monkeypatch
    ):
        from dsbf.engine import worker
        from dsbf.model import Note

        def returns_failure(check_, context: CheckContext) -> CheckResult:
            if check_.id != "null_share":
                return run_check(check_, context)
            return CheckResult(
                check_id=check_.id,
                check_version=check_.version,
                status=CheckStatus.ERROR,
                notes=(Note(code="failed", message="The check reported a failure."),),
                timing=Timing(wall_seconds=0.0),
            )

        monkeypatch.setattr(worker, "run_check", returns_failure)

        run = run_plan(TOYS)

        assert statuses(run) == {
            "null_share": CheckStatus.ERROR,
            "numeric_spread": CheckStatus.OK,
            "quality_summary": CheckStatus.SKIPPED,
        }
        assert run.health.counts.ok == 1
        assert run.health.counts.error == 1
        assert "The check reported a failure." in run.render_health()


class TestEligibilityAgreesWithThePlan:
    def test_the_check_reads_the_columns_the_plan_said(self):
        dataset = make_dataset()
        profile = profile_dataset(dataset)
        seen: list[tuple[str, ...]] = []

        def reader(ctx, params):
            seen.append(ctx.columns)

        shape_of = declare(
            "shape_of",
            applies_to=SemanticTypes("numeric"),
            requires=Requires(profile=["histogram"]),
            function=reader,
        )
        plan = plan_for([shape_of], profile)

        run = execute(plan, dataset, profile, dsbf_version="0.0.0")

        # `single` has one value, so no histogram, and the plan left it out.
        assert plan.get("shape_of").expected.columns == ("amount",)
        assert seen == [("amount",)]
        assert run.results["shape_of"].status is CheckStatus.OK

    def test_a_missing_dataset_feature_is_not_applicable_and_not_called(self):
        dataset = load(pl.DataFrame({"a": [1, 2, 3]}))
        profile = profile_dataset(dataset).model_copy(update={"duplicate_rows": None})
        called = []

        def dupes(ctx, params):
            called.append(True)

        declared = declare(
            "dupes",
            applies_to=None,
            requires=Requires(profile=["duplicate_rows"]),
            function=dupes,
        )
        plan = plan_for([declared], profile)

        run = execute(plan, dataset, profile, dsbf_version="0.0.0")

        assert plan.get("dupes").expected.status is Expected.NOT_APPLICABLE
        assert not called
        result = run.results["dupes"]
        assert result.status is CheckStatus.NOT_APPLICABLE
        assert result.notes[0].code == "profile_feature_missing"
        assert result.notes[0].message == plan.get("dupes").expected.reason
