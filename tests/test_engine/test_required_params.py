"""A check that requires a parameter is not applicable until the parameter is set."""

import pytest

from dsbf.engine import (
    CheckDefinitionError,
    Registry,
    Requires,
    Selection,
    SemanticTypes,
    check,
    make_plan,
)
from dsbf.engine.eligibility import eligibility
from dsbf.engine.expectation import Expected
from dsbf.model import CheckParams, Dimension
from dsbf.profiling import profile_dataset
from tests.test_engine.toys import make_dataset

pytestmark = pytest.mark.unit


class NeedsParams(CheckParams):
    label: str | None = None
    columns: dict[str, int] = {}


def declare(*required: str):
    @check(
        id="needy",
        version="1.0.0",
        kind="analysis",
        applies_to=SemanticTypes("numeric"),
        dimensions=[Dimension.DISTRIBUTION],
        params=NeedsParams,
        requires=Requires(params=required),
    )
    def declared(ctx, params):
        """A check."""

    return declared


def test_an_unset_parameter_makes_the_check_not_applicable():
    profile = profile_dataset(make_dataset())

    found = eligibility(declare("label").spec, profile, params={"label": None})

    assert not found.applicable
    assert found.code == "no_label"
    assert found.reason == "The `label` parameter is not set, which needy requires."


@pytest.mark.parametrize("value", [None, {}])
def test_an_empty_mapping_is_unset(value):
    found = eligibility(
        declare("columns").spec,
        profile_dataset(make_dataset()),
        params={"columns": value},
    )

    assert found.code == "no_columns"


def test_a_set_parameter_leaves_the_check_applicable():
    found = eligibility(
        declare("label").spec, profile_dataset(make_dataset()), params={"label": "y"}
    )

    assert found.applicable


def test_a_parameter_the_check_does_not_declare_is_a_definition_error():
    with pytest.raises(CheckDefinitionError, match="does not declare: nope"):
        declare("nope")


def test_the_plan_expects_not_applicable_until_the_parameter_is_given():
    registry = Registry()
    registry.register(declare("label"))
    profile = profile_dataset(make_dataset())
    selection = Selection(packs=[], include=["needy"])

    without = make_plan(registry, selection, profile=profile)
    given = make_plan(
        registry, selection, params={"needy": {"label": "y"}}, profile=profile
    )

    assert without.entries[0].expected.status is Expected.NOT_APPLICABLE
    assert given.entries[0].expected.status is Expected.WILL_RUN
