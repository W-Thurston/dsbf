# tests/test_model/test_evidence.py
"""Evidence cannot carry a value that serialises to invalid JSON."""

import json
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from dsbf.model import AssumptionCheck, Evidence, Interval, SamplingInfo

pytestmark = pytest.mark.unit


def evidence(**overrides):
    fields = {"metrics": {}, "n_used": 10, "n_total": 10, "method": "pearson_r"}
    return Evidence.model_validate({**fields, **overrides})


def strict_loads(text):
    """Parse JSON the way a strict consumer would: NaN and Infinity are errors."""

    def refuse(token):
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(text, parse_constant=refuse)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        np.nan,
        np.float64("inf"),
        np.float32("nan"),
        pd.NaT,
    ],
)
def test_undefined_metrics_become_null_and_are_flagged(value):
    result = evidence(metrics={"ok": 1.5, "bad": value})

    assert result.metrics == {"ok": 1.5, "bad": None}
    assert result.non_finite_metrics == ("bad",)
    assert strict_loads(result.model_dump_json())["metrics"]["bad"] is None


def test_the_flag_lists_every_undefined_metric_in_sorted_order():
    result = evidence(
        metrics={"z": float("nan"), "a": float("inf"), "m": 2},
        non_finite_metrics=("m",),
    )

    assert result.non_finite_metrics == ("a", "m", "z")


def test_a_clean_evidence_has_an_empty_flag():
    assert evidence(metrics={"r": 0.5}).non_finite_metrics == ()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3, 3),
        (2.5, 2.5),
        (True, True),
        ("text", "text"),
        (None, None),
        (np.int64(7), 7),
        (np.int8(-1), -1),
        (np.float64(0.25), 0.25),
        (np.float32(0.5), 0.5),
        (np.bool_(True), True),
        (np.str_("s"), "s"),
    ],
)
def test_metric_values_keep_their_python_type(value, expected):
    stored = evidence(metrics={"m": value}).metrics["m"]

    assert stored == expected
    assert type(stored) is type(expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (pd.Timestamp("2020-01-02 03:04:05"), "2020-01-02T03:04:05"),
        (pd.Timestamp("2020-01-02", tz="UTC"), "2020-01-02T00:00:00+00:00"),
        (datetime(2020, 1, 2, tzinfo=UTC), "2020-01-02T00:00:00+00:00"),
        (date(2020, 1, 2), "2020-01-02"),
        (
            np.datetime64("2020-01-02T03:04:05").astype("datetime64[us]").item(),
            "2020-01-02T03:04:05",
        ),
    ],
)
def test_timestamps_become_iso_strings_and_survive_json(value, expected):
    result = evidence(metrics={"first_seen": value})

    assert result.metrics["first_seen"] == expected
    restored = Evidence.model_validate_json(result.model_dump_json())
    assert restored == result


def test_a_stored_nan_token_is_repaired_when_read_back():
    stored = (
        '{"metrics": {"a": NaN, "b": Infinity, "c": -Infinity, "d": 1},'
        ' "n_used": 3, "n_total": 4, "method": "m"}'
    )

    result = Evidence.model_validate_json(stored)

    assert result.metrics == {"a": None, "b": None, "c": None, "d": 1}
    assert result.non_finite_metrics == ("a", "b", "c")


def test_metrics_are_json_serialisable_for_every_supported_value():
    result = evidence(
        metrics={
            "int": np.int64(1),
            "float": np.float64(1.5),
            "flag": np.bool_(False),
            "when": pd.Timestamp("2020-01-01"),
            "nan": float("nan"),
            "inf": float("inf"),
            "text": "x",
            "missing": None,
        }
    )

    strict_loads(result.model_dump_json())
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_a_metric_that_is_not_a_scalar_is_rejected():
    with pytest.raises(ValidationError):
        evidence(metrics={"values": [1, 2, 3]})


def test_the_metric_name_may_be_awkward():
    names = ["a|b", "[/x]", "", "日本語"]

    result = evidence(metrics={name: float("nan") for name in names})

    assert set(result.non_finite_metrics) == set(names)
    assert Evidence.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "counts",
    [{"n_used": 11, "n_total": 10}, {"n_used": -1}, {"n_total": -1}],
)
def test_counts_are_non_negative_and_used_is_within_total(counts):
    with pytest.raises(ValidationError):
        evidence(**counts)


def test_the_method_is_required_and_non_empty():
    with pytest.raises(ValidationError):
        evidence(method="")
    with pytest.raises(ValidationError):
        Evidence.model_validate({"metrics": {}, "n_used": 1, "n_total": 1})


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        evidence(sampling=None)


def test_a_full_evidence_round_trips():
    full = evidence(
        metrics={"r": 0.42, "p": 0.003, "n_pairs": 98, "tied": False, "label": "weak"},
        n_used=98,
        n_total=120,
        assumptions=(
            AssumptionCheck(name="monotonic", satisfied=True, method="visual"),
            AssumptionCheck(name="no_outliers", satisfied=None, detail="too few rows"),
        ),
        interval=Interval(low=0.25, high=0.57, level=0.95, method="fisher_z"),
        sampled=SamplingInfo(
            strategy="random", original_rows=120, sampled_rows=100, seed=42
        ),
        details_ref="artifacts/corr.parquet",
    )

    restored = Evidence.model_validate_json(full.model_dump_json())

    assert restored == full
    assert restored.interval is not None and restored.interval.level == 0.95


def test_evidence_is_immutable():
    with pytest.raises(ValidationError):
        evidence().n_used = 1  # type: ignore[misc]


def test_a_validated_instance_is_accepted_as_is():
    first = evidence(metrics={"x": float("nan")})

    assert Evidence.model_validate(first) == first
    assert Evidence.model_validate(first.model_dump()) == first
