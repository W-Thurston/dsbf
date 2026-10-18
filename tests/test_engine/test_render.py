"""``format_plan`` lays a plan out for a terminal."""

import pytest

from dsbf.engine import Registry, Requires, Selection, check, format_plan, make_plan
from dsbf.engine.check import SemanticTypes
from dsbf.model import Dimension

pytestmark = pytest.mark.unit


def declare(check_id: str, **spec):
    @check(
        id=check_id,
        version="1.0.0",
        kind="rule",
        applies_to=SemanticTypes(),
        dimensions=[Dimension.VALIDITY],
        **spec,
    )
    def declared(ctx, params):
        """A check."""

    return declared


def test_a_required_check_that_was_not_selected_is_marked_as_added():
    registry = Registry(
        [
            declare("base", packs=["lower"]),
            declare("top", requires=Requires(checks=["base"]), packs=["upper"]),
        ]
    )

    text = format_plan(make_plan(registry, Selection(packs="upper")))

    assert "added: required by top" in text


def test_checks_are_grouped_by_level():
    registry = Registry(
        [declare("base"), declare("top", requires=Requires(checks=["base"]))]
    )

    lines = format_plan(make_plan(registry)).splitlines()

    assert lines.index("level 0") < lines.index("level 1")


def test_a_plan_without_a_profile_says_columns_were_not_assessed():
    plan = make_plan(Registry([declare("only")]))

    assert "columns not assessed (no profile)" in format_plan(plan)


def test_the_dataset_names_the_header():
    plan = make_plan(Registry([declare("only")]))

    assert format_plan(plan, dataset="titanic (891 × 15)").startswith(
        "Plan for titanic (891 × 15): 1 check"
    )


def test_an_empty_plan_shows_its_note():
    text = format_plan(make_plan(Registry()))

    assert "nothing to run" in text
    assert "⚠ The selection matched no checks" in text
