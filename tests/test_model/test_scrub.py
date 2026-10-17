# tests/test_model/test_scrub.py
"""json_safe turns library values into plain JSON values, or says why it can't."""

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from dsbf.model.scrub import json_safe

pytestmark = pytest.mark.unit


def test_plain_values_pass_through():
    value = {"a": [1, 2.5, "x", True, None], "b": {"c": False}}

    assert json_safe(value) == (value, False)


def test_numpy_scalars_become_python_scalars():
    result, undefined = json_safe(
        {"i": np.int64(3), "f": np.float32(0.5), "b": np.bool_(True)}
    )

    assert result == {"i": 3, "f": 0.5, "b": True}
    assert {type(v) for v in result.values()} == {int, float, bool}
    assert undefined is False


def test_dates_become_iso_strings():
    result, _ = json_safe(
        [datetime(2020, 1, 2, 3, 4, tzinfo=UTC), date(2020, 1, 2), pd.Timestamp("2020")]
    )

    assert result == ["2020-01-02T03:04:00+00:00", "2020-01-02", "2020-01-01T00:00:00"]


def test_tuples_become_lists():
    assert json_safe((1, (2, 3))) == ([1, [2, 3]], False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -np.inf, pd.NaT])
def test_undefined_numbers_become_null_and_are_flagged(value):
    assert json_safe(value) == (None, True)
    assert json_safe({"k": [1, value]}) == ({"k": [1, None]}, True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), pd.NaT])
def test_strict_mode_rejects_undefined_numbers(value):
    with pytest.raises(ValueError, match="finite"):
        json_safe({"k": [value]}, null_non_finite=False)


@pytest.mark.parametrize("value", [object(), {1, 2}, b"x", {1: "a"}])
def test_values_that_are_not_json_are_rejected(value):
    with pytest.raises(ValueError):
        json_safe(value)
