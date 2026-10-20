"""The ``zeros`` check against ``detect_zeros``."""

from __future__ import annotations

from tests.golden.migration.harness import Difference, Result


def compare_zeros(legacy: Result, core: Result) -> list[Difference]:
    """Per column: if it is read, its zero count, share and flag, and its finding."""
    old = legacy["result"]["data"]
    columns = core["output"]["columns"]
    found: list[Difference] = []
    for column in sorted(set(old["zero_counts"]) | set(columns)):
        if column not in columns:
            found.append(Difference(column, "column_read", "legacy only"))
            continue
        if column not in old["zero_counts"]:
            found.append(Difference(column, "column_read", "core only"))
            continue
        new = columns[column]
        pairs = (
            ("zero_count", old["zero_counts"][column], new["zero_count"]),
            ("flagged", old["zero_flags"][column], new["flagged"]),
        )
        found += [
            Difference(column, kind, f"legacy {was}, core {now}")
            for kind, was, now in pairs
            if was != now
        ]
        was, now = old["zero_percentages"][column], new["zero_share"]
        if abs(was - now) > 1e-9:
            found.append(
                Difference(column, "zero_share", f"legacy {was:.4f}, core {now:.4f}")
            )

    was_levels = _legacy_levels(legacy["result"].get("guidance") or {})
    now_levels = {f["subject"]["columns"][0]: f["severity"] for f in core["findings"]}
    for column in sorted(set(was_levels) | set(now_levels)):
        was_level, now_level = was_levels.get(column), now_levels.get(column)
        if was_level is None:
            found.append(Difference(column, "finding", f"core only ({now_level})"))
        elif now_level is None:
            found.append(Difference(column, "finding", f"legacy only ({was_level})"))
        elif was_level != now_level:
            found.append(
                Difference(column, "severity", f"legacy {was_level}, core {now_level}")
            )
    return found


def _legacy_levels(guidance: dict) -> dict[str, str]:
    """The strongest level of the blurbs the task raised for each column."""
    levels: dict[str, str] = {}
    for column, phases in guidance.items():
        blurbs = [b for phase in phases.values() for b in phase]
        if blurbs:
            levels[column] = (
                "warn" if any(b["level"] == "warn" for b in blurbs) else "info"
            )
    return levels
