"""The missingness and string structure checks against the tasks they replace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.golden.migration.findings import columns_read, compare_levels
from tests.golden.migration.harness import Difference, Result

# What the encoded-column task called a kind, in the check's words.
_ENCODED_KINDS = {"high_entropy": "opaque_token"}
# The task's anomaly types, and the core output field that carries each.
_ANOMALIES = {
    "mixed_case": ("mixed_case", "affected_groups", "groups"),
    "leading_trailing_whitespace": ("edge_whitespace", "affected_count", "rows"),
    "invisible_characters": ("invisible_characters", "affected_count", "rows"),
    "length_outliers": ("length_outliers", "affected_count", "rows"),
}


def _core_columns(core: Result) -> Mapping[str, Mapping[str, Any]]:
    # A check with nothing to read ended not applicable and has no output.
    return (core["output"] or {}).get("columns", {})


def compare_missingness_mechanism(legacy: Result, core: Result) -> list[Difference]:
    """The columns analysed, their null counts, whether the nulls follow other columns.

    The task said a column was consistent with MAR when a correlation or a group
    difference was significant; the check says it depends on the observed columns when a
    test is flagged after adjustment.
    """
    data = legacy["result"]["data"]
    columns = _core_columns(core)
    found = columns_read(data, columns)
    for name in sorted(set(data) & set(columns)):
        was, now = data[name], columns[name]
        if was["n_missing"] != now["n_missing"]:
            found.append(
                Difference(
                    name,
                    "n_missing",
                    f"legacy {was['n_missing']}, core {now['n_missing']}",
                )
            )
        follows = "mar" in was["mechanism_assessment"]["consistent_with"]
        depends = now["assessment"] == "depends_on_observed"
        if follows != depends:
            found.append(
                Difference(
                    name,
                    "dependence",
                    f"legacy {'MAR' if follows else 'no MAR evidence'}, "
                    f"core {now['assessment']}",
                )
            )
    if legacy["result"]["summary"].get("little_mcar_test"):
        found.append(
            Difference("", "little_mcar_test", "legacy ran it, core has no such test")
        )
    return found + compare_levels(legacy, core)


def compare_mixed_types(legacy: Result, core: Result) -> list[Difference]:
    """The columns flagged, then the findings."""
    return columns_read(legacy["result"]["data"], _core_columns(core)) + compare_levels(
        legacy, core
    )


def compare_encoded_columns(legacy: Result, core: Result) -> list[Difference]:
    """The columns flagged and the kind each is, then the findings."""
    data = legacy["result"]["data"]
    columns = _core_columns(core)
    found = columns_read(data, columns)
    for name in sorted(set(data) & set(columns)):
        was = _ENCODED_KINDS.get(data[name]["match_type"], data[name]["match_type"])
        if was != columns[name]["kind"]:
            found.append(
                Difference(name, "kind", f"legacy {was}, core {columns[name]['kind']}")
            )
    return found + compare_levels(legacy, core)


def compare_regex_violations(legacy: Result, core: Result) -> list[Difference]:
    """The columns with violations and how many, then the findings."""
    was = (legacy["result"]["summary"] or {}).get("violations", {})
    now = {
        rule["column"]: rule
        for rule in (core["output"] or {}).get("rules", ())
        if rule["violations"]
    }
    found = columns_read(was, now)
    for name in sorted(set(was) & set(now)):
        if was[name]["num_violations"] != now[name]["violations"]:
            found.append(
                Difference(
                    name,
                    "violations",
                    f"legacy {was[name]['num_violations']}, "
                    f"core {now[name]['violations']}",
                )
            )
    return found + compare_levels(legacy, core)


def compare_string_anomalies(legacy: Result, core: Result) -> list[Difference]:
    """Each column's anomalies by type and their counts, then the findings."""
    data = legacy["result"]["data"]
    columns = _core_columns(core)
    found: list[Difference] = []
    for name in sorted(set(data) | set(columns)):
        was = {f["type"]: f for f in data.get(name, ())}
        now = columns.get(name, {})
        for kind, (field, old_key, new_key) in _ANOMALIES.items():
            before, after = was.get(kind), now.get(field)
            if before is None and after is None:
                continue
            if before is None:
                found.append(Difference(name, kind, "core only"))
            elif after is None:
                found.append(Difference(name, kind, "legacy only"))
            elif before[old_key] != after[new_key]:
                found.append(
                    Difference(
                        name,
                        kind,
                        f"legacy {before[old_key]}, core {after[new_key]}",
                    )
                )
    return found + compare_levels(legacy, core)
