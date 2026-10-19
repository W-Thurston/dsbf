"""Running a check yields a valid CheckResult, whatever kind of check it is."""

from typing import Any

import polars as pl
import pytest

from dsbf.engine import (
    CheckContext,
    CheckContractError,
    CheckOutputModel,
    NoParams,
    SemanticTypes,
    check,
    run_check,
)
from dsbf.model import CheckResult, CheckStatus, Dimension, Severity
from dsbf.profile import profile_dataset
from tests.test_engine.toys import (
    make_dataset,
    null_share,
    numeric_spread,
    quality_summary,
    run_toys,
)

pytestmark = pytest.mark.unit


def declare(function, **overrides: Any):
    fields: dict[str, Any] = {
        "id": "demo",
        "version": "1.0.0",
        "kind": "analysis",
        "applies_to": SemanticTypes("numeric"),
        "dimensions": [Dimension.VALIDITY],
    }
    return check(**{**fields, **overrides})(function)


def declare_with(**overrides):
    def decorate(function):
        return declare(function, **overrides)

    return decorate


def run(declared, dataset=None, **raw_params):
    dataset = dataset or make_dataset()
    context = CheckContext(
        declared.spec,
        dataset,
        profile_dataset(dataset),
        declared.resolve_params(raw_params),
    )
    return run_check(declared, context)


class TestToyChecks:
    def test_each_kind_yields_a_valid_result(self):
        results = run_toys()

        assert list(results) == ["null_share", "numeric_spread", "quality_summary"]
        for result in results.values():
            assert result.status is CheckStatus.OK
            assert result.timing.wall_seconds >= 0 and result.timing.started_at
            assert CheckResult.model_validate_json(result.model_dump_json()) == result

    def test_the_rule_check_reports_from_the_profile(self):
        result = run_toys()["null_share"]

        assert [f.subject.columns for f in result.findings] == [
            ("amount",),
            ("single",),
            ("empty",),
        ]
        assert result.output is None
        assert result.params == {}

    def test_the_analysis_check_reads_values_and_returns_typed_output(self):
        result = run_toys()["numeric_spread"]

        assert result.output is not None
        assert result.output["columns_read"] == 2
        assert result.output["std"]["single"] is None
        assert result.output["std"]["amount"] == pytest.approx(14.6, abs=1)
        assert result.params == {"min_std": 0.5}
        assert result.findings == ()

    def test_the_synthesis_check_reads_the_checks_before_it(self):
        result = run_toys()["quality_summary"]

        assert result.output == {"findings_seen": 3}
        assert [f.code for f in result.findings] == ["findings_rolled_up"]
        assert result.findings[0].subject.kind == "dataset"

    def test_a_parameter_changes_what_is_found_and_is_recorded(self):
        result = run(numeric_spread, min_std=100)

        assert result.params == {"min_std": 100.0}
        assert [f.code for f in result.findings] == ["low_spread"]
        assert result.findings[0].dimension is Dimension.DISTRIBUTION

    def test_a_sample_is_recorded_on_every_finding(self):
        dataset = make_dataset(rows=60, strategy="random", seed=3)

        result = run(null_share, dataset)

        assert result.findings
        assert {f.evidence.sampled for f in result.findings} == {dataset.sampling}


class TestNotApplicable:
    def test_no_eligible_column_skips_the_function_and_says_why(self):
        called = []

        @check(
            id="demo",
            version="1.0.0",
            kind="analysis",
            applies_to=SemanticTypes("datetime"),
            dimensions=[Dimension.TEMPORAL],
        )
        def never(ctx, params):
            called.append(True)

        result = run(never, make_dataset(pl.DataFrame({"x": [1.5, 2.5, 3.5] * 20})))

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert not called
        (note,) = result.notes
        assert note.code == "no_eligible_columns"
        assert "datetime" in note.message and "demo" in note.message
        assert result.findings == ()

    def test_a_dataset_level_check_always_runs(self):
        result = run(quality_summary, make_dataset(pl.DataFrame({"x": range(30)})))

        assert result.status is CheckStatus.OK


class TestOutput:
    def test_an_undefined_number_is_nulled_and_named(self):
        class Ratio(CheckOutputModel):
            ratio: float
            inner: dict[str, float]

        @declare_with(output=Ratio)
        def ratio(ctx, params):
            return ctx.output(Ratio(ratio=float("nan"), inner={"a": float("inf")}))

        result = run(ratio)

        assert result.output == {"ratio": None, "inner": {"a": None}}
        assert result.non_finite_outputs == ("inner", "ratio")

    def test_notes_on_an_ok_result_are_kept(self):
        @declare_with()
        def noisy(ctx, params):
            ctx.note("capped", "Looked at 3 of 9 columns.", severity=Severity.NOTICE)

        result = run(noisy)

        assert result.status is CheckStatus.OK
        assert [(n.code, n.severity) for n in result.notes] == [
            ("capped", Severity.NOTICE)
        ]


class TestContract:
    def test_an_exception_in_the_check_propagates_to_the_executor(self):
        @declare_with()
        def broken(ctx, params):
            raise ZeroDivisionError("boom")

        with pytest.raises(ZeroDivisionError, match="boom"):
            run(broken)

    def test_a_check_must_return_what_the_context_built(self):
        @declare_with()
        def wrong(ctx, params):
            return {"std": 1}

        with pytest.raises(CheckContractError, match="returned dict"):
            run(wrong)

    def test_a_context_belongs_to_one_check(self):
        dataset = make_dataset()
        context = CheckContext(
            null_share.spec, dataset, profile_dataset(dataset), NoParams()
        )

        with pytest.raises(CheckContractError, match="built for null_share, not"):
            run_check(numeric_spread, context)


class TestStop:
    """A check ends in a status other than ``ok`` without raising."""

    @pytest.mark.parametrize(
        "status",
        [
            CheckStatus.NOT_APPLICABLE,
            CheckStatus.UNAVAILABLE,
            CheckStatus.SKIPPED,
            CheckStatus.ERROR,
        ],
    )
    def test_each_allowed_status_is_returned_with_its_note(self, status):
        @declare_with()
        def stops(ctx, params):
            return ctx.stop(
                status,
                "why",
                "Because.",
                severity=Severity.ERROR,
                hint="Do this.",
                exception_type="ValueError",
                detail="trace",
            )

        result = run(stops)

        assert result.status is status
        (note,) = result.notes
        assert (note.code, note.message, note.severity) == (
            "why",
            "Because.",
            Severity.ERROR,
        )
        assert (note.hint, note.exception_type, note.detail) == (
            "Do this.",
            "ValueError",
            "trace",
        )
        assert result.output is None

    def test_findings_made_before_stopping_are_dropped_and_notes_kept(self):
        @declare_with()
        def half_done(ctx, params):
            ctx.finding(**finding_fields())
            ctx.note("capped", "Looked at 10 columns.")
            return ctx.stop(CheckStatus.ERROR, "gave_up", "Gave up.")

        result = run(half_done)

        assert result.findings == ()
        assert [n.code for n in result.notes] == ["capped", "gave_up"]

    @pytest.mark.parametrize("status", [CheckStatus.OK, CheckStatus.TIMEOUT])
    def test_ok_and_timeout_are_not_the_checks_to_choose(self, status):
        @declare_with()
        def overreach(ctx, params):
            return ctx.stop(status, "why", "Because.")

        with pytest.raises(CheckContractError, match="cannot stop with status"):
            run(overreach)

    def test_a_check_that_does_not_stop_stays_ok(self):
        @declare_with()
        def fine(ctx, params):
            return ctx.output()

        assert run(fine).status is CheckStatus.OK


def finding_fields():
    from dsbf.model import Confidence, Evidence, Subject

    return {
        "code": "x",
        "subject": Subject.column("amount"),
        "severity": Severity.INFO,
        "confidence": Confidence.HIGH,
        "title": "t",
        "message": "m",
        "evidence": Evidence(metrics={}, n_used=1, n_total=1, method="m"),
    }
