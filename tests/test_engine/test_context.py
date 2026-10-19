"""A check sees only the columns its declaration selects, and builds findings through
the context."""

import numpy as np
import polars as pl
import pytest

from dsbf.data import load
from dsbf.engine import (
    CheckContext,
    CheckContractError,
    CheckOutputModel,
)
from dsbf.model import (
    Confidence,
    Dimension,
    Evidence,
    Severity,
    Subject,
)
from dsbf.profiling import profile_dataset
from tests.test_engine.toys import (
    EXPECTED_TYPES,
    SpreadOutput,
    make_dataset,
    null_share,
    numeric_spread,
    quality_summary,
)

pytestmark = pytest.mark.unit


def context_for(toy, dataset=None, **kwargs):
    dataset = dataset or make_dataset()
    return CheckContext(
        toy.spec, dataset, profile_dataset(dataset), toy.resolve_params(), **kwargs
    )


def evidence(n=10):
    return Evidence(metrics={"n": n}, n_used=n, n_total=n, method="count")


def finding_kwargs(**overrides):
    return {
        "code": "demo",
        "subject": Subject.column("amount"),
        "severity": Severity.INFO,
        "confidence": Confidence.HIGH,
        "title": "Demo",
        "message": "A demo.",
        "evidence": evidence(),
    } | overrides


def test_the_fixture_has_every_semantic_type():
    profile = profile_dataset(make_dataset())

    assert {c.name: c.semantic_type for c in profile.columns} == EXPECTED_TYPES


class TestColumnScope:
    def test_columns_are_those_the_check_applies_to_in_dataset_order(self):
        assert context_for(numeric_spread).columns == ("amount", "single")

    def test_no_names_selects_every_column(self):
        assert context_for(null_share).columns == tuple(EXPECTED_TYPES)

    def test_a_dataset_level_check_has_no_columns(self):
        assert context_for(quality_summary).columns == ()

    def test_values_outside_the_scope_are_refused(self):
        context = context_for(numeric_spread)

        assert context.values("amount").dtype == np.float64
        for outside in ("colour", "when", "no_such_column"):
            with pytest.raises(CheckContractError, match="outside that scope"):
                context.values(outside)

    def test_the_profile_view_holds_only_the_scoped_columns(self):
        view = context_for(numeric_spread).profile

        assert [c.name for c in view.columns] == ["amount", "single"]
        assert view.column("amount").semantic_type == "numeric"
        with pytest.raises(CheckContractError, match="No column 'colour'"):
            view.column("colour")
        assert view.n_rows == 200

    def test_null_pairs_name_only_scoped_columns(self):
        frame = pl.DataFrame(
            {
                "a": [None if i % 4 == 0 else float(i) for i in range(80)],
                "b": [None if i % 4 == 0 else float(i * i) for i in range(80)],
                "c": [None if i % 4 == 0 else "xyz"[i % 3] for i in range(80)],
            }
        )
        dataset = load(frame)
        profile = profile_dataset(dataset)
        assert profile.missingness and len(profile.missingness.pairs) == 3

        context = CheckContext(numeric_spread.spec, dataset, profile)

        assert [(p.a, p.b) for p in context.profile.null_pairs] == [("a", "b")]

    def test_a_profile_of_another_dataset_is_refused(self):
        other = make_dataset(pl.DataFrame({"z": range(5)}))

        with pytest.raises(ValueError, match="different columns"):
            CheckContext(numeric_spread.spec, make_dataset(), profile_dataset(other))
        with pytest.raises(ValueError, match="50 rows, the dataset 200"):
            CheckContext(
                numeric_spread.spec,
                make_dataset(),
                profile_dataset(make_dataset(rows=50)),
            )


class TestKinds:
    @pytest.mark.parametrize("toy", [null_share, quality_summary])
    def test_only_an_analysis_check_reads_values(self, toy):
        with pytest.raises(CheckContractError, match="does not read values"):
            context_for(toy).values("amount")


class TestUpstreamResults:
    def test_a_declared_dependency_is_readable_and_absent_when_it_did_not_run(self):
        context = context_for(quality_summary)

        assert context.result("numeric_spread") is None
        assert context.result("null_share") is None

    def test_an_undeclared_dependency_is_refused(self):
        with pytest.raises(CheckContractError, match="did not declare 'other'"):
            context_for(quality_summary).result("other")


class TestFindings:
    def test_a_finding_takes_its_identity_from_the_check_and_subject(self):
        context = context_for(numeric_spread)

        finding = context.finding(**finding_kwargs(dimension=Dimension.VALIDITY))

        assert (finding.check_id, finding.check_version) == ("numeric_spread", "1.2.0")
        assert finding.id == type(finding).make_id(
            "numeric_spread", Subject.column("amount"), "demo"
        )
        assert context.findings == (finding,)

    def test_the_dimension_defaults_to_the_only_one_declared(self):
        assert context_for(null_share).finding(**finding_kwargs()).dimension is (
            Dimension.COMPLETENESS
        )

    def test_several_dimensions_require_a_choice(self):
        with pytest.raises(CheckContractError, match="declares 2 dimensions"):
            context_for(numeric_spread).finding(**finding_kwargs())

    def test_an_undeclared_dimension_is_refused(self):
        with pytest.raises(CheckContractError, match="did not declare dimension"):
            context_for(null_share).finding(
                **finding_kwargs(dimension=Dimension.LEAKAGE)
            )

    def test_a_subject_outside_the_scope_is_refused(self):
        context = context_for(numeric_spread)

        with pytest.raises(CheckContractError, match="outside that scope"):
            context.finding(
                **finding_kwargs(
                    subject=Subject.column("colour"), dimension=Dimension.VALIDITY
                )
            )
        with pytest.raises(CheckContractError, match="outside that scope"):
            context.finding(
                **finding_kwargs(
                    subject=Subject.column_pair("amount", "colour"),
                    dimension=Dimension.VALIDITY,
                )
            )
        assert context.findings == ()

    def test_a_dataset_check_may_only_use_the_dataset_subject(self):
        context = context_for(quality_summary)

        context.finding(**finding_kwargs(subject=Subject.dataset()))
        with pytest.raises(CheckContractError, match="applies to no columns"):
            context.finding(**finding_kwargs(code="other"))

    def test_a_subject_and_code_are_reported_once(self):
        context = context_for(null_share)
        context.finding(**finding_kwargs())

        with pytest.raises(CheckContractError, match="already reported 'demo'"):
            context.finding(**finding_kwargs())
        context.finding(**finding_kwargs(code="other"))
        assert len(context.findings) == 2

    def test_a_finding_about_a_sample_carries_the_sampling(self):
        dataset = make_dataset(rows=50, strategy="random", seed=7)
        context = context_for(null_share, dataset)

        finding = context.finding(**finding_kwargs())

        assert finding.evidence.sampled == dataset.sampling
        assert dataset.sampling and dataset.sampling.sampled_rows == 50

    def test_a_finding_about_the_whole_dataset_has_no_sampling(self):
        assert (
            context_for(null_share).finding(**finding_kwargs()).evidence.sampled is None
        )


class TestNotes:
    def test_a_note_is_recorded_with_its_hint(self):
        context = context_for(null_share)

        note = context.note("capped", "Used 10 of 200 pairs.", hint="Raise max_pairs.")

        assert context.notes == (note,)
        assert (note.severity, note.hint) == (Severity.INFO, "Raise max_pairs.")


class TestOutput:
    def test_an_output_model_becomes_plain_json_values(self):
        output = context_for(numeric_spread).output(
            SpreadOutput(std={"a": 1.5, "b": None}, columns_read=2)
        )

        assert output.data == {"std": {"a": 1.5, "b": None}, "columns_read": 2}

    def test_no_output_is_allowed(self):
        assert context_for(null_share).output().data is None

    def test_an_output_without_a_declared_model_is_refused(self):
        class Stray(CheckOutputModel):
            n: int

        with pytest.raises(CheckContractError, match="declares no output model"):
            context_for(null_share).output(Stray(n=1))

    def test_an_output_of_another_model_is_refused(self):
        class Stray(CheckOutputModel):
            n: int

        with pytest.raises(CheckContractError, match="declares output SpreadOutput"):
            context_for(numeric_spread).output(Stray(n=1))
