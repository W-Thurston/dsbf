"""Unit tests for the golden snapshot normaliser (not the golden runs)."""

from __future__ import annotations

import json

import pytest

from tests.golden.normalise import (
    INLINE_CHARS,
    MAX_ENTRIES,
    build_snapshot,
    canonical,
    compact_json,
    digest,
    flatten_guidance,
    normalise_task,
    reduce_value,
    render_snapshot,
    round_sig,
    scrub_text,
)

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


def test_scrub_text_replaces_longest_root_first():
    roots = ["/tmp/run", "/tmp/run/out"]
    assert scrub_text("/tmp/run/out/figs/a.png and /tmp/run/x", roots) == (
        "<run>/figs/a.png and <run>/x"
    )


def test_reduce_value_keeps_small_structures_whole():
    small = {"a": [1, 2, 3], "b": {"c": 1.5}}
    assert reduce_value(small) == small


def test_reduce_value_summarises_wide_collections_with_a_digest():
    wide = {f"k{i}": "x" * 60 for i in range(MAX_ENTRIES + 1)}
    reduced = reduce_value(wide)
    assert reduced == {"_len": MAX_ENTRIES + 1, "_digest": digest(wide)}
    changed = {**wide, "k0": "y" * 60}
    assert reduce_value(changed)["_digest"] != reduced["_digest"]


def test_reduce_value_descends_into_a_large_dict_with_few_keys():
    value = {
        "small": {"a": 1},
        "big": [{"row": "x" * 80, "i": i} for i in range(MAX_ENTRIES - 1)],
    }
    assert len(compact_json(value)) > INLINE_CHARS
    reduced = reduce_value(value)
    assert reduced["small"] == {"a": 1}
    assert set(reduced["big"]) == {"_len", "_digest"}


def test_flatten_guidance_finds_items_in_any_nesting_and_drops_bodies():
    guidance = {
        "age": {
            "eda": [
                {
                    "phase": "eda",
                    "column": "age",
                    "level": "warn",
                    "title": "T1",
                    "body": "long",
                }
            ],
            "ml": [{"phase": "ml", "column": "age", "level": "info", "title": "T2"}],
        },
        "__dataset__": [{"column": None, "level": "error", "title": "T0"}],
    }
    flat = flatten_guidance(guidance)
    assert {(g["level"], g["title"]) for g in flat} == {
        ("error", "T0"),
        ("warn", "T1"),
        ("info", "T2"),
    }
    assert all("body" not in g for g in flat)
    assert flatten_guidance(None) == []
    assert flatten_guidance({}) == []


def test_flatten_guidance_order_does_not_depend_on_input_order():
    a = {"x": [{"level": "warn", "title": "A"}], "y": [{"level": "info", "title": "B"}]}
    b = {"y": [{"level": "info", "title": "B"}], "x": [{"level": "warn", "title": "A"}]}
    assert flatten_guidance(a) == flatten_guidance(b)


def test_volatile_tasks_keep_only_their_status():
    result = {"status": "success", "data": {"top": [{"duration_sec": 1.0}]}}
    entry = normalise_task("identify_bottleneck_tasks", result, [])
    assert entry == {"status": "success"}


def test_normalise_task_scrubs_paths_and_rounds_floats():
    result = {
        "status": "success",
        "summary": {"message": "ok", "score": 0.12345678901},
        "data": {"plot": "/tmp/run/out/figs/a.png"},
        "guidance": {
            "c": [{"level": "info", "title": "t", "phase": "eda", "column": "c"}]
        },
    }
    entry = normalise_task("some_task", result, ["/tmp/run/out", "/tmp/run"])
    assert entry["summary"]["score"] == 0.123457
    assert entry["data"] == {"plot": "<run>/figs/a.png"}
    assert entry["guidance"][0]["title"] == "t"
    assert entry["error"] is None


def test_rendered_snapshot_is_independent_of_dict_order_and_float_noise():
    def snapshot(order, noise):
        results = {
            "b": {"status": "success", "summary": {"v": 0.1 + noise}},
            "a": {"status": "success", "summary": {"v": 2.0}},
        }
        report = {"results": {k: results[k] for k in order}, "top_ml_risks": []}
        return render_snapshot(build_snapshot("d", report, {"shape": [1, 2]}, []))

    assert snapshot(["a", "b"], 0.0) == snapshot(["b", "a"], 1e-15)


def test_rendered_snapshot_is_valid_json_with_a_trailing_newline():
    report = {
        "results": {"a": {"status": "success", "summary": {"v": float("nan")}}},
        "top_ml_risks": [],
    }
    text = render_snapshot(build_snapshot("d", report, {"shape": [1, 2]}, []))
    assert text.endswith("}\n")
    assert json.loads(text)["tasks"]["a"]["summary"] == {"v": "NaN"}
