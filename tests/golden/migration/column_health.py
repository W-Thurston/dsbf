"""The column health checks against the legacy tasks they replace."""

from __future__ import annotations

from tests.golden.migration.findings import compare_levels
from tests.golden.migration.harness import Difference, Result

_TOLERANCE = 1e-9


def _flagged(subject_set: set[str], old: set[str], new: set[str]) -> list[Difference]:
    return [
        Difference(column, "flagged", "legacy only" if column in old else "core only")
        for column in sorted(subject_set)
        if (column in old) != (column in new)
    ]


def compare_constant_columns(legacy: Result, core: Result) -> list[Difference]:
    old = set(legacy["result"]["data"]["constant_columns"])
    new = set(core["output"]["constant_columns"])
    return _flagged(old | new, old, new) + compare_levels(legacy, core)


def compare_high_cardinality(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]
    columns = core["output"]["columns"]
    new = {name for name, column in columns.items() if column["high"]}
    found = _flagged(set(old) | new, set(old), new)
    for column in sorted(set(old) & new):
        if old[column] != columns[column]["distinct"]:
            found.append(
                Difference(
                    column,
                    "n_unique",
                    f"legacy {old[column]}, core {columns[column]['distinct']}",
                )
            )
    return found + compare_levels(legacy, core)


def compare_id_columns(legacy: Result, core: Result) -> list[Difference]:
    old = set(legacy["result"]["data"])
    new = set(core["output"]["id_columns"])
    return _flagged(old | new, old, new) + compare_levels(legacy, core)


def compare_single_dominant_value(legacy: Result, core: Result) -> list[Difference]:
    """Per column the check reads: its mode, the mode's share and the flag.

    The task read every column. Those the check does not read matter only if the task
    raised a finding about them, which the level comparison reports.
    """
    old = legacy["result"]["data"]
    columns = core["output"]["columns"]
    found: list[Difference] = []
    for column in sorted(set(old) & set(columns)):
        was, now = old[column], columns[column]
        if was["mode"] != now["top_value"]:
            found.append(
                Difference(
                    column,
                    "top_value",
                    f"legacy {was['mode']}, core {now['top_value']}",
                )
            )
        if abs(was["mode_proportion"] - now["top_share"]) > 1e-4:
            found.append(
                Difference(
                    column,
                    "top_share",
                    f"legacy {was['mode_proportion']:.4f}, core {now['top_share']:.4f}",
                )
            )
        if (was["mode_proportion"] >= 0.95) != now["flagged"]:
            found.append(
                Difference(column, "flagged", f"core flagged {now['flagged']}")
            )
    return found + compare_levels(legacy, core, columns=set(old))


def compare_boolean_fields(legacy: Result, core: Result) -> list[Difference]:
    """The task's only findings are about imbalance, which ``dominant_value`` raises."""
    examined = set(legacy["result"]["data"])
    return compare_levels(legacy, core, columns=examined)


def compare_near_zero_variance(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]["low_variance_columns"]
    columns = core["output"]["columns"]
    new = {name for name, column in columns.items() if column["low"]}
    found = _flagged(set(old) | new, set(old), new)
    for column in sorted(set(old) & new):
        was, now = old[column], columns[column]["variance"]
        if abs(was - now) > 1e-7:
            found.append(
                Difference(column, "variance", f"legacy {was:.3e}, core {now:.3e}")
            )
    return found + compare_levels(legacy, core)


def compare_duplicate_rows(legacy: Result, core: Result) -> list[Difference]:
    was = legacy["result"]["data"]["duplicate_count"]
    now = core["output"]["duplicate_rows"]
    found = []
    if was != now:
        found.append(Difference("", "duplicate_rows", f"legacy {was}, core {now}"))
    return found + compare_levels(legacy, core)


def compare_duplicate_columns(legacy: Result, core: Result) -> list[Difference]:
    """Legacy pairs against the pairs inside the core groups, and the levels."""
    old = {tuple(pair) for pair in legacy["result"]["data"]["duplicate_column_pairs"]}
    new = {
        (a, b)
        for group in core["output"]["groups"]
        for i, a in enumerate(group)
        for b in group[i + 1 :]
    }
    found = [
        Difference(f"{a}, {b}", "pair", "legacy only" if (a, b) in old else "core only")
        for a, b in sorted(old ^ new)
    ]
    return found + compare_levels(legacy, core)


def compare_null_rate(legacy: Result, core: Result) -> list[Difference]:
    data = legacy["result"]["data"]
    columns = core["output"]["columns"]
    found: list[Difference] = []
    for column in sorted(set(data["null_counts"]) | set(columns)):
        if column not in columns:
            found.append(Difference(column, "column_read", "legacy only"))
            continue
        if column not in data["null_counts"]:
            found.append(Difference(column, "column_read", "core only"))
            continue
        was, now = data["null_counts"][column], columns[column]["null_count"]
        if was != now:
            found.append(Difference(column, "null_count", f"legacy {was}, core {now}"))
        was_share, now_share = (
            data["null_percentages"][column],
            columns[column]["null_share"],
        )
        if abs(was_share - now_share) > _TOLERANCE:
            found.append(
                Difference(
                    column,
                    "null_share",
                    f"legacy {was_share:.4f}, core {now_share:.4f}",
                )
            )
    mostly = {name for name, column in columns.items() if column["mostly_missing"]}
    old_mostly = set(data["high_null_columns"])
    found += [
        Difference(column, "mostly_missing", f"legacy {column in old_mostly}")
        for column in sorted(old_mostly ^ mostly)
    ]
    return found + compare_levels(legacy, core)
