"""Limits reject values that cannot mean a time limit."""

import math

import pytest

from dsbf.engine import Limits

pytestmark = pytest.mark.unit


def test_the_default_limits_nothing():
    limits = Limits()

    assert limits.check_timeout_seconds is None
    assert limits.run_budget_seconds is None


@pytest.mark.parametrize("name", ["check_timeout_seconds", "run_budget_seconds"])
@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan])
def test_a_limit_must_be_positive_and_finite(name, value):
    with pytest.raises(ValueError, match=name):
        Limits(**{name: value})


@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_the_grace_must_be_finite_and_not_negative(value):
    with pytest.raises(ValueError, match="cancel_grace_seconds"):
        Limits(cancel_grace_seconds=value)


def test_no_grace_is_allowed():
    assert Limits(cancel_grace_seconds=0).cancel_grace_seconds == 0
