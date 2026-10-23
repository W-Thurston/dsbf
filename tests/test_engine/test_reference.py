"""A reference dataset reaches only the checks that require one."""

import polars as pl
import pytest

from dsbf.data import load
from dsbf.engine import (
    CheckContext,
    CheckContractError,
    Reference,
    Registry,
    Requires,
    Selection,
    SemanticTypes,
    check,
    execute,
    make_plan,
)
from dsbf.engine.cache import CacheSession
from dsbf.engine.eligibility import eligibility
from dsbf.model import CheckStatus, Dimension
from dsbf.profiling import profile_dataset
from tests.test_engine.process_toys import py_reference_rows
from tests.test_engine.test_cache import MemoryCache
from tests.test_engine.toys import build_frame, make_dataset

pytestmark = pytest.mark.unit


def declare(check_id: str, *, reference: bool, function=None):
    @check(
        id=check_id,
        version="1.0.0",
        kind="analysis",
        applies_to=SemanticTypes("numeric"),
        dimensions=[Dimension.DISTRIBUTION],
        requires=Requires(reference=reference),
    )
    def declared(ctx, params):
        """A check."""
        if function:
            return function(ctx)

    return declared


def a_reference(rows: int = 50) -> Reference:
    dataset = load(pl.DataFrame({"amount": [float(i) for i in range(rows)]}))
    return Reference.of_data(dataset, profile_dataset(dataset))


class TestEligibility:
    def test_a_check_that_requires_a_reference_is_not_applicable_without_one(self):
        profile = profile_dataset(make_dataset())
        spec = declare("compare", reference=True).spec

        found = eligibility(spec, profile)

        assert not found.applicable
        assert found.code == "no_reference"
        assert found.reason == (
            "No reference dataset was given, which compare compares with."
        )

    def test_it_is_applicable_with_one(self):
        profile = profile_dataset(make_dataset())
        found = eligibility(
            declare("compare", reference=True).spec, profile, has_reference=True
        )

        assert found.applicable and found.columns == ("amount", "single")

    def test_a_check_that_does_not_require_one_never_asks(self):
        profile = profile_dataset(make_dataset())

        assert eligibility(declare("plain", reference=False).spec, profile).applicable

    def test_the_plan_expects_what_the_run_will_do(self):
        dataset = make_dataset()
        profile = profile_dataset(dataset)
        registry = Registry([declare("compare", reference=True)])
        selection = Selection(entry_points=False)

        without = make_plan(registry, selection, profile=profile)
        with_one = make_plan(registry, selection, profile=profile, has_reference=True)

        assert without.entries[0].expected.status.value == "not_applicable"
        assert with_one.entries[0].expected.status.value == "will_run"


class TestContext:
    def test_a_check_that_declared_it_reads_the_reference(self):
        spec = declare("compare", reference=True).spec
        dataset = make_dataset()
        context = CheckContext(
            spec, dataset, profile_dataset(dataset), reference=a_reference()
        )

        assert context.reference.profile.n_rows == 50

    def test_a_check_that_did_not_declare_it_cannot_read_it(self):
        spec = declare("plain", reference=False).spec
        dataset = make_dataset()
        context = CheckContext(
            spec, dataset, profile_dataset(dataset), reference=a_reference()
        )

        with pytest.raises(CheckContractError, match="requires.reference"):
            context.reference

    def test_a_run_gives_the_reference_only_to_the_check_that_requires_it(self):
        seen: dict[str, object] = {}

        def peek(ctx):
            seen["rows"] = ctx.reference.profile.n_rows

        def try_peek(ctx):
            try:
                ctx.reference
            except CheckContractError:
                seen["blocked"] = True

        dataset = make_dataset()
        profile = profile_dataset(dataset)
        plan = make_plan(
            Registry(
                [
                    declare("reads", reference=True, function=peek),
                    declare("tries", reference=False, function=try_peek),
                ]
            ),
            Selection(entry_points=False),
            profile=profile,
            has_reference=True,
        )

        run = execute(
            plan,
            dataset,
            profile,
            dsbf_version="test",
            reference=a_reference(),
            workers=1,
        )

        assert seen == {"rows": 50, "blocked": True}
        assert {r.status for r in run.results.values()} == {CheckStatus.OK}

    def test_a_worker_process_gets_it_too(self):
        dataset = make_dataset()
        profile = profile_dataset(dataset)
        plan = make_plan(
            Registry([py_reference_rows]),
            Selection(entry_points=False),
            profile=profile,
            has_reference=True,
        )

        run = execute(
            plan, dataset, profile, dsbf_version="test", reference=a_reference(70)
        )

        result = run.results["py_reference_rows"]
        assert result.status is CheckStatus.OK
        assert result.findings[0].evidence.metrics == {"rows": 70}


class TestCacheKey:
    def key(self, check_id: str, reference: Reference | None, reference_check: bool):
        dataset = make_dataset()
        profile = profile_dataset(dataset)
        session = CacheSession(MemoryCache(), dataset, profile, reference)
        plan = make_plan(
            Registry([declare(check_id, reference=reference_check)]),
            Selection(entry_points=False),
            profile=profile,
            has_reference=reference is not None,
        )
        entry = plan.entries[0]
        return session._key(entry, {}).digest

    def test_a_different_reference_is_a_different_key(self):
        assert self.key("compare", a_reference(50), True) != self.key(
            "compare", a_reference(60), True
        )

    def test_the_same_reference_is_the_same_key(self):
        assert self.key("compare", a_reference(50), True) == self.key(
            "compare", a_reference(50), True
        )

    def test_a_check_that_does_not_read_the_reference_is_unaffected(self):
        assert self.key("plain", a_reference(50), False) == self.key(
            "plain", a_reference(60), False
        )


def test_a_reference_made_from_a_sample_keeps_the_full_data_identity():
    dataset = load(build_frame(), name="big").sample(40, strategy="random", seed=3)
    reference = Reference.of_data(dataset, profile_dataset(dataset))

    assert reference.kind == "data"
    assert reference.n_rows == 200 and reference.profile.n_rows == 40
    assert reference.sampling is not None
    assert reference.fingerprint == dataset.fingerprint
