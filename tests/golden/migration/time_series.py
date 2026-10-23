"""The datetime and time-series checks against the tasks they replace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.golden.migration.findings import columns_read, compare_levels
from tests.golden.migration.harness import Difference, Result


def _columns(core: Result) -> Mapping[str, Mapping[str, Any]]:
    # A check with nothing to read ended not applicable and has no output.
    return (core["output"] or {}).get("columns", {})


def _label(strength: float) -> str:
    return "strong" if strength >= 0.7 else "moderate" if strength >= 0.4 else "weak"


def compare_temporal_gaps(legacy: Result, core: Result) -> list[Difference]:
    """The columns judged, each column's gap count, then the findings.

    The task counted intervals over twice the mode of the rounded day gaps; the check
    counts steps over 1.5 times the dominant step of the distinct values and calls a
    column with no dominant step irregular.
    """
    data = legacy["result"]["data"]
    columns = _columns(core)
    found = columns_read(data, columns)
    for name in sorted(set(data) & set(columns)):
        was, now = data[name]["large_gap_count"], columns[name]["n_gaps"]
        if was != (now or 0) or (now is None) != (columns[name]["frequency"] is None):
            found.append(
                Difference(
                    name,
                    "gaps",
                    f"legacy {was}, core {'irregular' if now is None else now}",
                )
            )
    return found + compare_levels(legacy, core)


def compare_datetime_consistency(legacy: Result, core: Result) -> list[Difference]:
    """The columns checked and the values that did not read."""
    data = legacy["result"]["data"]
    columns = _columns(core)
    found = columns_read(data, columns)
    for name in sorted(set(data) & set(columns)):
        was, now = data[name]["num_invalid"], columns[name]["unparsed"]
        if was != now:
            found.append(Difference(name, "unparsed", f"legacy {was}, core {now}"))
    return found


def compare_datetime_features(legacy: Result, core: Result) -> list[Difference]:
    """The columns described and the features listed for each, then the findings."""
    data = legacy["result"]["data"]
    columns = _columns(core)
    found = columns_read(data, columns)
    for name in sorted(set(data) & set(columns)):
        was = set(data[name]["recommended_features"])
        now = set(columns[name]["features"])
        if was != now:
            found.append(
                Difference(
                    name,
                    "features",
                    f"legacy only {sorted(was - now)}, core only {sorted(now - was)}",
                )
            )
    return found + compare_levels(legacy, core)


def compare_stationarity(legacy: Result, core: Result) -> list[Difference]:
    """The columns tested and the joint reading of the two tests for each."""
    data = legacy["result"]["data"]
    columns = _columns(core)
    found = columns_read(data, columns)
    for name in sorted(set(data) & set(columns)):
        was, now = data[name]["assessment"], columns[name]["assessment"]
        if was != now:
            found.append(Difference(name, "assessment", f"legacy {was}, core {now}"))
    return found


def compare_autocorrelation(legacy: Result, core: Result) -> list[Difference]:
    """The columns analysed and whether each has an autocorrelation worth reporting.

    The task listed every lag outside the band; the check flags those of 0.2 or more.
    """
    data = legacy["result"]["data"]
    columns = _columns(core)
    found = columns_read(data, columns)
    for name in sorted(set(data) & set(columns)):
        was = bool(data[name]["acf_significant_lags"])
        now = bool(columns[name]["flagged_acf"])
        if was != now:
            found.append(
                Difference(
                    name,
                    "autocorrelated",
                    f"legacy {'yes' if was else 'no'}, core {'yes' if now else 'no'}",
                )
            )
    return found


def compare_seasonal_decomposition(legacy: Result, core: Result) -> list[Difference]:
    """The columns decomposed, the period and the strength labels of each.

    The task needed a frequency to infer a period and decomposed nothing without one.
    """
    data = legacy["result"]["data"]
    columns = _columns(core)
    if not data and columns:
        return [Difference("", "columns", f"legacy 0, core {len(columns)}")]
    found = columns_read(data, columns)
    for name in sorted(set(data) & set(columns)):
        was, now = data[name], columns[name]
        if was["seasonal_period"] != now["period"]:
            found.append(
                Difference(
                    name,
                    "period",
                    f"legacy {was['seasonal_period']}, core {now['period']}",
                )
            )
        for part in ("trend", "seasonal"):
            before = _label(was[f"{part}_strength"])
            after = _label(now[f"{part}_strength"])
            if before != after:
                found.append(Difference(name, part, f"legacy {before}, core {after}"))
    return found
