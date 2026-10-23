"""The columns a check reads are the same in the plan, the context and the run."""

import polars as pl
import pytest

from dsbf.data import load
from dsbf.engine import CheckContext, Requires, SemanticTypes, check
from dsbf.engine.eligibility import eligibility
from dsbf.engine.errors import CheckContractError
from dsbf.model import Dimension
from dsbf.profiling import profile_dataset
from tests.test_engine.toys import make_dataset

pytestmark = pytest.mark.unit


def declare(applies_to, **spec):
    @check(
        id="demo",
        version="1.0.0",
        kind="rule",
        applies_to=applies_to,
        dimensions=[Dimension.VALIDITY],
        **spec,
    )
    def declared(ctx, params):
        """A check."""

    return declared


def test_columns_are_those_of_the_type_that_have_the_required_features():
    dataset = make_dataset()
    profile = profile_dataset(dataset)
    shape_of = declare(
        SemanticTypes("numeric"), requires=Requires(profile=["histogram"])
    )

    found = eligibility(shape_of.spec, profile)
    context = CheckContext(shape_of.spec, dataset, profile)

    assert found.columns == ("amount",) and found.applicable
    assert context.columns == ("amount",)
    assert [c.name for c in context.profile.columns] == ["amount"]
    assert context.eligibility == found


def test_a_check_that_reads_no_columns_is_applicable_with_none():
    found = eligibility(declare(None).spec, profile_dataset(make_dataset()))

    assert found.columns is None and found.applicable and found.code is None


def test_a_synthesis_check_reads_every_column_but_no_values():
    dataset = make_dataset()
    profile = profile_dataset(dataset)

    @check(
        id="advice",
        version="1.0.0",
        kind="synthesis",
        applies_to=None,
        dimensions=[Dimension.USABILITY],
        after=["null_share"],
    )
    def advice(ctx, params):
        """A synthesis check."""

    found = eligibility(advice.spec, profile)
    context = CheckContext(advice.spec, dataset, profile)

    assert found.columns == tuple(c.name for c in profile.columns)
    assert found.columns is not None
    assert context.columns == found.columns
    assert [c.name for c in context.profile.columns] == list(found.columns)
    with pytest.raises(CheckContractError, match="does not read values"):
        context.values("amount")


def test_nothing_of_the_type_names_the_type():
    profile = profile_dataset(load(pl.DataFrame({"x": ["a", "b", "c"]})))

    found = eligibility(declare(SemanticTypes("datetime")).spec, profile)

    assert not found.applicable
    assert found.code == "no_eligible_columns"
    assert found.columns == ()
    assert found.reason == "No column is of a type demo applies to (datetime)."


def test_a_missing_dataset_feature_names_the_feature():
    profile = profile_dataset(load(pl.DataFrame({"a": [1, 2, 3]})))
    profile = profile.model_copy(update={"missingness": None})

    needs_missingness = declare(None, requires=Requires(profile=["missingness"]))

    found = eligibility(needs_missingness.spec, profile)

    assert not found.applicable
    assert found.code == "profile_feature_missing"
    assert found.columns is None
    assert found.reason == "The profile has no missingness, which demo requires."
