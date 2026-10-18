"""A plan validates the parameters it is given, and reports every mistake at once."""

import pytest
from pydantic import Field

from dsbf.engine import PlanError, Registry, Selection, check, make_plan
from dsbf.engine.check import SemanticTypes
from dsbf.model import CheckParams, Dimension

pytestmark = pytest.mark.unit


class SpreadParams(CheckParams):
    min_std: float = Field(0.5, ge=0, description="Spread below this is reported.")
    method: str = "sample"


@check(
    id="spread",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.DISTRIBUTION],
    params=SpreadParams,
    packs=["core"],
)
def spread(ctx, params):
    """Spread."""


@check(
    id="plain",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.VALIDITY],
    packs=["core"],
)
def plain(ctx, params):
    """Takes no parameters."""


@pytest.fixture
def registry() -> Registry:
    return Registry([spread, plain])


def problems(registry: Registry, params, selection: Selection | None = None):
    with pytest.raises(PlanError) as raised:
        make_plan(registry, selection, params=params)
    return raised.value.problems


class TestResolvedParams:
    def test_an_entry_carries_the_defaults_when_nothing_is_given(self, registry):
        plan = make_plan(registry)

        assert plan.get("spread").params == SpreadParams()

    def test_given_values_override_the_defaults(self, registry):
        plan = make_plan(registry, params={"spread": {"min_std": 2}})

        assert plan.get("spread").params == SpreadParams(min_std=2.0, method="sample")

    def test_parameters_of_a_registered_check_outside_the_plan_are_allowed(
        self, registry
    ):
        plan = make_plan(
            registry, Selection(exclude="spread"), params={"spread": {"min_std": 1}}
        )

        assert plan.ids == ("plain",)


class TestMistakes:
    def test_a_misspelled_key_names_the_check_key_and_allowed_keys(self, registry):
        (problem,) = problems(registry, {"spread": {"min_stdd": 1}})

        assert problem == (
            "spread: unknown parameter 'min_stdd'; allowed: min_std, method "
            "(did you mean 'min_std'?)"
        )

    def test_a_check_without_parameters_says_so(self, registry):
        (problem,) = problems(registry, {"plain": {"threshold": 1}})

        assert (
            problem
            == "plain: unknown parameter 'threshold'; this check takes no parameters"
        )

    def test_a_value_outside_its_bounds_gives_the_reason_and_the_input(self, registry):
        (problem,) = problems(registry, {"spread": {"min_std": -1}})

        assert problem == (
            "spread: parameter 'min_std': Input should be greater than or equal to 0 "
            "(got -1)"
        )

    def test_a_value_of_the_wrong_type_is_reported(self, registry):
        (problem,) = problems(registry, {"spread": {"min_std": "wide"}})

        assert "parameter 'min_std'" in problem
        assert "'wide'" in problem

    def test_parameters_that_are_not_a_mapping_are_reported(self, registry):
        (problem,) = problems(registry, {"spread": 3})

        assert (
            problem == "spread: parameters must be a mapping of name to value, not int"
        )

    def test_an_unknown_check_id_suggests_the_closest(self, registry):
        (problem,) = problems(registry, {"spred": {"min_std": 1}})

        assert problem == "params names unknown check 'spred'; did you mean 'spread'?"

    def test_every_mistake_is_reported_together(self, registry):
        found = problems(
            registry,
            {"spread": {"min_stdd": 1, "method": 3}, "plain": {"x": 1}, "nope": {}},
        )

        assert len(found) == 4
        assert found[0].startswith("params names unknown check 'nope'")

    def test_a_selection_mistake_still_comes_first(self, registry):
        with pytest.raises(PlanError, match="unknown pack"):
            make_plan(registry, Selection(packs="cor"), params={"spread": {"x": 1}})
