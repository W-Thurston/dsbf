"""`reference_drift` against the three drift tasks, on a dataset and a perturbed copy.

The tasks compared the data with a reference and raised a blurb per flagged column;
the check does the same from the reference's profile. Their binning and thresholds
differ, so a numeric column is compared by whether it is flagged and how strongly, not
by the value of its index.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.golden.migration.findings import columns_read
from tests.golden.migration.harness import Difference, Result

# The change in a column's share of nulls from which `reference_drift` reports it; the
# task flagged a change over 0.3, so its own flag is not what is compared.
MISSING_SHIFT = 0.1
_LEGACY_LEVEL = {"low": "none", "moderate": "moderate", "high": "high"}
_TARGET_SEVERITY = {"none": None, "moderate": "warn", "significant": "error"}


def _output(core: Result) -> Mapping[str, Any]:
    return core["output"] or {}


def _columns_with(core: Result, code: str) -> set[str]:
    return {f["subject"]["columns"][0] for f in core["findings"] if f["code"] == code}


def compare_compare_with_reference_dataset(
    legacy: Result, core: Result
) -> list[Difference]:
    """The added and dropped columns, the type changes, then the shifts in nulls.

    The task also flagged a change in distinct count and in the minimum and maximum of
    a column; the check reads those as part of the column's distribution.
    """
    data = legacy["result"]["data"]
    output = _output(core)
    found: list[Difference] = []
    for was_key, now_key in (
        ("added_columns", "columns_added"),
        ("dropped_columns", "columns_dropped"),
    ):
        was, now = sorted(data[was_key]), sorted(output[now_key])
        if was != now:
            found.append(Difference("", was_key, f"legacy {was}, core {now}"))
    for was_set, now_set, kind in (
        (set(data["type_mismatches"]), _columns_with(core, "type_changed"), "type"),
        (
            {
                column
                for column, change in data["field_changes"].items()
                if change.get("missing_pct_diff", 0) >= MISSING_SHIFT
            },
            _columns_with(core, "missingness_shift"),
            "missingness_shift",
        ),
    ):
        found += [
            Difference(
                column, kind, "legacy only" if column in was_set else "core only"
            )
            for column in sorted(was_set ^ now_set)
        ]
    return found


def compare_detect_feature_drift(legacy: Result, core: Result) -> list[Difference]:
    """The columns read, then for each whether it is flagged and how strongly."""
    data = legacy["result"]["data"]
    columns = _output(core).get("columns", {})
    read = {c for c, v in data.items() if v["type"] in ("numerical", "categorical")}
    compared = {c for c, v in columns.items() if v["compared"] != "none"}
    found = columns_read(read, compared)
    for column in sorted(read & compared):
        was = _LEGACY_LEVEL[data[column]["severity"]]
        now = columns[column]["drift"]
        if (was == "none") != (now == "none"):
            who, level = ("legacy", was) if was != "none" else ("core", now)
            found.append(Difference(column, "flagged", f"{who} only ({level})"))
        elif was != now:
            found.append(Difference(column, "level", f"legacy {was}, core {now}"))
    return found


def compare_detect_target_drift(legacy: Result, core: Result) -> list[Difference]:
    """Whether the target is flagged, and the strength of the drift."""
    rating = legacy["result"]["data"]["drift_rating"]
    was = _TARGET_SEVERITY[rating]
    drift = [f for f in core["findings"] if f["code"] == "target_drift"]
    now = drift[0]["severity"] if drift else None
    if was == now:
        return []
    if was is None or now is None:
        return [
            Difference(
                "target",
                "flagged",
                f"legacy {rating}, core {'drift' if now else 'none'}",
            )
        ]
    return [Difference("target", "level", f"legacy {rating}, core {now}")]
