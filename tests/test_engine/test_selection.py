"""A selection picks checks by pack, depth, cost, dimension, tag and id."""

import pytest

from dsbf.engine import (
    Check,
    Depth,
    Pack,
    PlanError,
    Registry,
    Selection,
    check,
    select,
)
from dsbf.engine.check import SemanticTypes
from dsbf.model import Cost, Dimension, Stage

pytestmark = pytest.mark.unit


def make_check(
    check_id: str,
    *,
    cost: Cost = Cost.CHEAP,
    packs: tuple[str, ...] = ("core",),
    dimensions: tuple[Dimension, ...] = (Dimension.VALIDITY,),
    tags: tuple[str, ...] = (),
    stage: Stage = Stage.ANY,
) -> Check:
    @check(
        id=check_id,
        version="1.0.0",
        kind="rule",
        applies_to=SemanticTypes(),
        dimensions=dimensions,
        cost=cost,
        packs=packs,
        tags=tags,
        stage=stage,
    )
    def declared(ctx, params):
        """A check for selection tests."""

    return declared


@pytest.fixture
def registry() -> Registry:
    return Registry(
        [
            make_check("null_rate"),
            make_check("zeros", dimensions=(Dimension.COMPLETENESS,), tags=("basic",)),
            make_check("outliers", cost=Cost.MODERATE, tags=("basic", "stats")),
            make_check(
                "fuzzy_duplicates",
                cost=Cost.HEAVY,
                dimensions=(Dimension.REDUNDANCY,),
            ),
            make_check(
                "seasonality",
                cost=Cost.MODERATE,
                packs=("timeseries",),
                dimensions=(Dimension.TEMPORAL,),
                stage=Stage.CLEANED,
            ),
            make_check("report_card", packs=(), stage=Stage.REPORT),
        ]
    )


def picked(registry: Registry, **selection) -> list[str]:
    return [c.id for c in select(registry, Selection(**selection))]


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        pytest.param(
            {},
            ["null_rate", "outliers", "report_card", "seasonality", "zeros"],
            id="the-standard-depth-is-the-default-and-stops-at-moderate",
        ),
        pytest.param(
            {"depth": "quick"},
            ["null_rate", "report_card", "zeros"],
            id="quick-keeps-cheap-checks",
        ),
        pytest.param(
            {"depth": "deep"},
            [
                "fuzzy_duplicates",
                "null_rate",
                "outliers",
                "report_card",
                "seasonality",
                "zeros",
            ],
            id="deep-allows-heavy-checks",
        ),
        pytest.param(
            {"depth": "deep", "max_cost": "cheap"},
            ["null_rate", "report_card", "zeros"],
            id="a-lower-max-cost-wins-over-the-depth",
        ),
        pytest.param(
            {"depth": "quick", "max_cost": "heavy"},
            ["null_rate", "report_card", "zeros"],
            id="a-higher-max-cost-does-not-raise-the-depth",
        ),
        pytest.param(
            {"packs": "timeseries"},
            ["seasonality"],
            id="one-pack",
        ),
        pytest.param(
            {"packs": ["core", "timeseries"], "depth": "quick"},
            ["null_rate", "zeros"],
            id="packs-then-depth",
        ),
        pytest.param(
            {"dimensions": ["completeness", "temporal"]},
            ["seasonality", "zeros"],
            id="a-check-matches-any-listed-dimension",
        ),
        pytest.param(
            {"tags": ["stats"]},
            ["outliers"],
            id="a-tag",
        ),
        pytest.param(
            {"tags": ["basic", "stats"]},
            ["outliers", "zeros"],
            id="a-check-matches-any-listed-tag",
        ),
        pytest.param(
            {"packs": "core", "tags": "basic", "dimensions": "completeness"},
            ["zeros"],
            id="filters-are-anded",
        ),
        pytest.param(
            {"exclude_stages": ["cleaned", "report"]},
            ["null_rate", "outliers", "zeros"],
            id="excluded-stages",
        ),
        pytest.param(
            {"include": "fuzzy_duplicates", "packs": "timeseries", "depth": "quick"},
            ["fuzzy_duplicates"],
            id="include-bypasses-pack-depth-and-filters",
        ),
        pytest.param(
            {"include": ["fuzzy_duplicates"], "tags": "basic", "depth": "deep"},
            ["fuzzy_duplicates", "outliers", "zeros"],
            id="include-adds-to-what-the-filters-keep",
        ),
        pytest.param(
            {"exclude": ["null_rate", "seasonality"]},
            ["outliers", "report_card", "zeros"],
            id="exclude-removes-by-id",
        ),
        pytest.param(
            {"packs": "core", "depth": "quick", "exclude": "zeros"},
            ["null_rate"],
            id="exclude-applies-after-packs-and-depth",
        ),
    ],
)
def test_selection_picks_checks(registry, selection, expected):
    assert picked(registry, **selection) == expected


def test_a_pack_with_no_checks_is_valid_and_selects_nothing():
    registry = Registry([make_check("null_rate")])
    registry.register_pack(Pack("empty", []))

    assert picked(registry, packs="empty") == []


def test_a_pack_object_and_a_spec_share_one_pack_name():
    stray = make_check("stray", packs=())
    registry = Registry([make_check("null_rate")])
    registry.register_pack(Pack("core", [stray]))

    assert picked(registry, packs="core") == ["null_rate", "stray"]


def test_selecting_from_an_empty_registry_is_empty():
    assert picked(Registry()) == []


@pytest.mark.parametrize(
    ("selection", "messages"),
    [
        pytest.param(
            {"packs": "timeseris"},
            ["unknown pack 'timeseris'; did you mean 'timeseries'?"],
            id="a-misspelled-pack-suggests-the-close-name",
        ),
        pytest.param(
            {"packs": "healthcare"},
            ["unknown pack 'healthcare'; registered packs: core, timeseries"],
            id="an-unknown-pack-lists-the-known-ones",
        ),
        pytest.param(
            {"include": "null_rat"},
            ["include names unknown check 'null_rat'; did you mean 'null_rate'?"],
            id="an-unknown-include",
        ),
        pytest.param(
            {"exclude": "zero"},
            ["exclude names unknown check 'zero'; did you mean 'zeros'?"],
            id="an-unknown-exclude",
        ),
        pytest.param(
            {"include": "zeros", "exclude": "zeros"},
            ["'zeros' is both included and excluded"],
            id="included-and-excluded",
        ),
        pytest.param(
            {"packs": "nope", "include": "nada", "exclude": "zeros"},
            [
                "unknown pack 'nope'",
                "include names unknown check 'nada'",
            ],
            id="every-mistake-is-reported-together",
        ),
    ],
)
def test_a_selection_mistake_is_a_plan_error_naming_it(registry, selection, messages):
    with pytest.raises(PlanError) as raised:
        select(registry, Selection(**selection))

    assert len(raised.value.problems) == len(messages)
    for problem, message in zip(raised.value.problems, messages, strict=True):
        assert message in problem


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (
            {"depth": "thorough"},
            "'thorough' is not a depth; choose from quick, standard, deep",
        ),
        ({"max_cost": "free"}, "'free' is not a cost"),
        ({"dimensions": ["quality"]}, "'quality' is not a dimension"),
        ({"exclude_stages": ["late"]}, "'late' is not a stage"),
        ({"packs": ["Core Pack"]}, "pack 'Core Pack' must be lower snake case"),
        ({"include": ["Null-Rate"]}, "include 'Null-Rate' must be lower snake case"),
    ],
)
def test_a_malformed_selection_fails_when_built(selection, message):
    with pytest.raises((PlanError, ValueError), match=message):
        Selection(**selection)


def test_a_selection_normalises_what_it_is_given():
    selection = Selection(
        packs="core",
        include=["a", "a"],
        depth="deep",
        max_cost=Cost.CHEAP,
        dimensions=["validity"],
        plugins="plugins/mine",
    )

    assert selection.packs == ("core",)
    assert selection.include == ("a",)
    assert selection.depth is Depth.DEEP
    assert selection.cost_ceiling is Cost.CHEAP
    assert selection.dimensions == (Dimension.VALIDITY,)
    assert selection.plugins == ("plugins/mine",)
    assert selection.entry_points is True
