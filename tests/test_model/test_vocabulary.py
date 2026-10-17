# tests/test_model/test_vocabulary.py
"""Severity, confidence and dimension are closed, ordered where ranking makes sense."""

import operator

import pytest

from dsbf.model import Confidence, Dimension, Severity

pytestmark = pytest.mark.unit


def test_severity_ranks_from_info_to_error():
    ordered = [Severity.INFO, Severity.NOTICE, Severity.WARN, Severity.ERROR]

    assert list(Severity) == ordered
    assert sorted(reversed(ordered)) == ordered
    assert Severity.ERROR > Severity.WARN >= Severity.WARN > Severity.NOTICE
    assert Severity.INFO < Severity.NOTICE <= Severity.NOTICE
    assert max(ordered) is Severity.ERROR


def test_confidence_ranks_from_low_to_high():
    assert Confidence.LOW < Confidence.MODERATE < Confidence.HIGH
    assert sorted([Confidence.HIGH, Confidence.LOW, Confidence.MODERATE]) == list(
        Confidence
    )


def test_different_vocabularies_do_not_compare():
    with pytest.raises(TypeError):
        operator.lt(Severity.INFO, Confidence.HIGH)


def test_members_are_their_plain_strings():
    assert Severity.WARN == "warn"
    assert Confidence("moderate") is Confidence.MODERATE
    assert f"{Dimension.LEAKAGE}" == "leakage"


def test_dimension_values_are_the_documented_set():
    assert {d.value for d in Dimension} == {
        "completeness",
        "validity",
        "usability",
        "redundancy",
        "leakage",
        "distribution",
        "relationship",
        "temporal",
    }


def test_unknown_values_are_rejected():
    with pytest.raises(ValueError):
        Severity("critical")
