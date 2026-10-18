"""A check's declaration is validated when its module is imported."""

import tomllib
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from dsbf.engine import (
    CheckDefinitionError,
    Extra,
    NoParams,
    Requires,
    SemanticTypes,
    check,
)
from dsbf.model import Cost, Dimension, SemanticType
from tests.test_engine.toys import SpreadParams, numeric_spread, quality_summary

pytestmark = pytest.mark.unit

VALID: dict[str, Any] = {
    "id": "demo",
    "version": "1.0.0",
    "kind": "rule",
    "applies_to": SemanticTypes(),
    "dimensions": [Dimension.VALIDITY],
}


def declare(**overrides: Any):
    return check(**{**VALID, **overrides})(lambda ctx, params: None)


def test_a_declaration_keeps_its_metadata_as_a_frozen_spec():
    spec = numeric_spread.spec

    assert (spec.id, spec.version, spec.kind) == ("numeric_spread", "1.2.0", "analysis")
    assert spec.applies_to == SemanticTypes("numeric")
    assert spec.dimensions == (Dimension.DISTRIBUTION, Dimension.VALIDITY)
    assert spec.params is SpreadParams
    assert spec.cost is Cost.MODERATE
    assert spec.packs == ("core",)
    assert spec.summary == "Standard deviation of each numeric column."
    with pytest.raises(AttributeError):
        spec.id = "other"  # type: ignore[misc]


def test_a_synthesis_check_lists_what_it_reads():
    spec = quality_summary.spec

    assert spec.upstream == ("null_share", "numeric_spread")
    assert spec.applies_to is None


def test_decorating_returns_a_callable_check():
    declared = declare()

    assert declared.id == "demo"
    assert declared(None, NoParams()) is None  # type: ignore[bad-argument-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": "NullRate"}, "id 'NullRate' must be lower snake case"),
        ({"id": "null-rate"}, "must be lower snake case"),
        ({"version": "1.0"}, "must be MAJOR.MINOR.PATCH"),
        ({"kind": "report"}, "kind 'report' must be one of rule, analysis, synthesis"),
        ({"params": dict}, "params must be a CheckParams class"),
        ({"output": dict}, "output must be a CheckOutputModel class"),
        ({"applies_to": "numeric"}, "applies_to is SemanticTypes"),
        ({"dimensions": []}, "declare at least one dimension"),
        ({"dimensions": ["quality"]}, "'quality' is not a valid Dimension"),
        ({"cost": "free"}, "'free' is not a valid Cost"),
        ({"extras": ["gpu"]}, "unknown extra 'gpu'; choose from ml, timeseries"),
        ({"packs": ["Core Pack"]}, "pack 'Core Pack' must be lower snake case"),
        ({"after": ["Bad"]}, "after 'Bad' must be lower snake case"),
        ({"after": ["demo"]}, "cannot depend on itself"),
        (
            {"requires": Requires(checks=["a"]), "after": ["a"]},
            "'a' is both required and in after",
        ),
        (
            {"kind": "synthesis", "applies_to": None},
            "a synthesis check names the checks it reads",
        ),
        (
            {"kind": "synthesis", "after": ["a"]},
            "a synthesis check reads other checks, not columns",
        ),
    ],
)
def test_a_bad_declaration_fails_at_definition_with_the_field_named(overrides, message):
    with pytest.raises(CheckDefinitionError, match=message):
        declare(**overrides)


def test_a_requirement_names_a_known_profile_feature_and_a_valid_check_id():
    with pytest.raises(
        CheckDefinitionError, match="unknown profile feature 'quantile'"
    ):
        Requires(profile=["quantile"])  # type: ignore[list-item]
    with pytest.raises(CheckDefinitionError, match="requires.checks 'Bad'"):
        Requires(checks=["Bad"])
    assert Requires(profile=["counts", "counts"]).profile == ("counts",)


def test_a_check_cannot_require_itself():
    with pytest.raises(CheckDefinitionError, match="cannot depend on itself"):
        declare(requires=Requires(checks=["demo"]))


def test_semantic_types_must_be_known():
    with pytest.raises(
        CheckDefinitionError, match="unknown semantic type 'continuous'"
    ):
        SemanticTypes("continuous")


def test_semantic_types_select_by_name_and_no_names_selects_everything():
    numeric = SemanticTypes("numeric", "boolean", "numeric")

    assert numeric.names == ("numeric", "boolean")
    assert numeric.selects("boolean") and not numeric.selects("text")
    assert all(SemanticTypes().selects(name) for name in get_args(SemanticType))


def test_repeated_names_are_collapsed():
    spec = declare(packs=["core", "core"], tags=["a", "a"]).spec

    assert (spec.packs, spec.tags) == (("core",), ("a",))


def test_parameters_resolve_against_the_declared_model():
    assert numeric_spread.resolve_params() == SpreadParams()
    assert numeric_spread.resolve_params({"min_std": 2}) == SpreadParams(min_std=2)
    with pytest.raises(ValidationError, match="min_sd"):
        numeric_spread.resolve_params({"min_sd": 2})
    with pytest.raises(ValidationError):
        numeric_spread.resolve_params({"min_std": -1})


def test_the_extras_a_check_may_name_are_the_projects_extras():
    pyproject = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text()
    )
    declared = set(pyproject["project"]["optional-dependencies"]) - {"all"}

    assert set(get_args(Extra)) == declared
