"""Unit tests for the golden snapshot normaliser (not the golden runs)."""

from __future__ import annotations

import json

import pytest

from tests.golden.normalise import canonical, render_snapshot, round_sig

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.281521129617477, 1.28152),
        (1.2815211296174769, 1.28152),
        (123456789.0, 123457000.0),
        (0.000123456789, 0.000123457),
        (-0.0, 0.0),
        (0.0, 0.0),
    ],
)
def test_round_sig_keeps_six_significant_figures(value, expected):
    assert round_sig(value) == expected


def test_round_sig_is_stable_across_last_digit_noise():
    assert round_sig(1.2815211296174769) == round_sig(1.281521129617477)


def test_round_sig_spells_out_non_finite_values():
    assert round_sig(float("nan")) == "NaN"
    assert round_sig(float("inf")) == "Infinity"
    assert round_sig(float("-inf")) == "-Infinity"


def test_canonical_drops_volatile_keys_at_any_depth():
    value = {
        "keep": 1,
        "approx_memory_MB": 3.2,
        "nested": {"duration_sec": 0.5, "column_memory_bytes": {"a": 1}, "n": 2},
        "start_time": "2026-10-16T11:30:07",
        "time_series_enabled": False,
    }
    assert canonical(value) == {
        "keep": 1,
        "nested": {"n": 2},
        "time_series_enabled": False,
    }


def test_canonical_sorts_keys_and_leaves_bools_and_ints_alone():
    assert list(canonical({"b": True, "a": 2})) == ["a", "b"]
    assert canonical({"a": True, "b": 3}) == {"a": True, "b": 3}


def test_rendered_snapshot_is_independent_of_dict_order_and_float_noise():
    def snapshot(order, noise):
        values = {"b": {"v": 0.1 + noise}, "a": {"v": 2.0}}
        return render_snapshot(canonical({k: values[k] for k in order}))

    assert snapshot(["a", "b"], 0.0) == snapshot(["b", "a"], 1e-15)


def test_rendered_snapshot_is_valid_json_with_a_trailing_newline():
    text = render_snapshot(canonical({"a": {"v": float("nan")}}))

    assert text.endswith("}\n")
    assert json.loads(text) == {"a": {"v": "NaN"}}
