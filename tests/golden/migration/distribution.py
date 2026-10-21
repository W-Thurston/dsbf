"""The shape, type and contract checks against the legacy tasks they replace."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from tests.golden.migration.findings import compare_levels
from tests.golden.migration.harness import Difference, Result

# The relative gap between the profile's bias-corrected skewness and the legacy
# task's biased one is sqrt(n(n-1))/(n-2) - 1, under 3% from 30 rows.
_SKEW_TOLERANCE = 0.03
_KURTOSIS_TOLERANCE = 1e-3  # the task rounded to four decimals
_ENTROPY_TOLERANCE = 1e-9

_TAILS = {
    "strongly_leptokurtic": "heavy",
    "mildly_leptokurtic": "mild",
    "platykurtic": "thin",
    "mesokurtic": "normal",
}
_DTYPES = {
    "bool": "Boolean",
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "category": "Categorical",
}


def _read(old: Collection[str], new: Collection[str]) -> list[Difference]:
    """The columns only one of them read."""
    return [
        Difference(
            column, "column_read", "legacy only" if column in old else "core only"
        )
        for column in sorted(set(old) ^ set(new))
    ]


def _only(core: Result, codes: Collection[str]) -> Result:
    """The core result with just the findings of the given codes."""
    return {**core, "findings": [f for f in core["findings"] if f["code"] in codes]}


def compare_skewness(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]
    columns = core["output"]["columns"]
    found = _read(old, columns)
    for column in sorted(set(old) & set(columns)):
        was, now = old[column], columns[column]["skewness"]
        if now is None:
            found.append(Difference(column, "skewness", f"legacy {was:.4f}, core none"))
        elif abs(was - now) > _SKEW_TOLERANCE * max(abs(was), 1e-3):
            found.append(
                Difference(column, "skewness", f"legacy {was:.4f}, core {now:.4f}")
            )
    return found + compare_levels(legacy, _only(core, {"skewed"}))


def compare_summarize_numeric(legacy: Result, core: Result) -> list[Difference]:
    """The task's two kinds of finding, which two checks raise between them."""
    return compare_levels(
        legacy, _only(core, {"near_zero_variance", "mean_median_gap"})
    )


def compare_kurtosis(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]
    columns = core["output"]["columns"]
    found = _read(old, columns)
    for column in sorted(set(old) & set(columns)):
        was, now = old[column], columns[column]
        if now["kurtosis"] is None:
            found.append(Difference(column, "kurtosis", "core none"))
            continue
        if abs(was["kurtosis"] - now["kurtosis"]) > _KURTOSIS_TOLERANCE:
            found.append(
                Difference(
                    column,
                    "kurtosis",
                    f"legacy {was['kurtosis']:.4f}, core {now['kurtosis']:.4f}",
                )
            )
        if _TAILS[was["classification"]] != now["tails"]:
            found.append(
                Difference(
                    column,
                    "tails",
                    f"legacy {_TAILS[was['classification']]}, core {now['tails']}",
                )
            )
    return found + compare_levels(legacy, core)


def compare_entropy(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]
    columns = core["output"]["columns"]
    found = _read(old, columns)
    for column in sorted(set(old) & set(columns)):
        was, now = old[column], columns[column]["entropy_bits"]
        if abs(was - now) > _ENTROPY_TOLERANCE:
            found.append(
                Difference(column, "entropy_bits", f"legacy {was:.6f}, core {now:.6f}")
            )
    return found


def compare_dtype_optimizations(legacy: Result, core: Result) -> list[Difference]:
    """Per column: whether a suggestion is made and to which type.

    The task also suggested float32 for every float column, which the core check
    drops; those are counted as one difference, not one per column. The estimated
    saving is not compared: the task read pandas memory, the check estimates it from
    widths.
    """
    old = legacy["result"]["data"]["suggestions"]
    new = core["output"]["suggestions"]
    float32 = {c for c, s in old.items() if s["suggested_dtype"] == "float32"}
    kept = {c: s for c, s in old.items() if c not in float32}
    found: list[Difference] = []
    if float32:
        found.append(Difference("", "float32_suggestions", f"legacy {len(float32)}"))
    for column in sorted(set(kept) | set(new)):
        if column not in new:
            found.append(Difference(column, "suggestion", "legacy only"))
        elif column not in kept:
            found.append(Difference(column, "suggestion", "core only"))
        else:
            was = _DTYPES[kept[column]["suggested_dtype"]]
            if was != new[column]["suggested_dtype"]:
                found.append(
                    Difference(
                        column,
                        "suggested_dtype",
                        f"legacy {was}, core {new[column]['suggested_dtype']}",
                    )
                )
    return found + _levels_without(legacy, core, float32)


def _levels_without(
    legacy: Result, core: Result, dropped: Collection[str]
) -> list[Difference]:
    """The level comparison, leaving out the columns whose suggestion was dropped."""
    return [d for d in compare_levels(legacy, core) if d.subject not in dropped]


def compare_sample_size(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]
    rules = core["output"]["rules"]
    found: list[Difference] = []
    for rule in sorted(set(old) | set(rules)):
        if rule not in rules or rule not in old:
            found.append(
                Difference(rule, "rule", "legacy only" if rule in old else "core only")
            )
            continue
        for key in ("required_rows", "adequate"):
            was, now = old[rule][key], rules[rule][key]
            if was != now:
                found.append(Difference(rule, key, f"legacy {was}, core {now}"))
    return found + compare_levels(legacy, core)


def compare_not_run(legacy: Result, core: Result) -> list[Difference]:
    """Tasks that need a setting the golden runs do not give: both end the same way."""
    return compare_levels(legacy, core)


def merge_core(results: dict[str, Any], checks: tuple[str, ...]) -> Result:
    """One core result for a task replaced by several checks.

    Its status is ``ok`` only if every check's is; its findings are all of theirs and
    its output is theirs by check id.
    """
    if len(checks) == 1:
        return results[checks[0]]
    statuses = [results[check]["status"] for check in checks]
    return {
        "status": "ok"
        if set(statuses) == {"ok"}
        else next(s for s in statuses if s != "ok"),
        "findings": [f for check in checks for f in results[check]["findings"]],
        "output": {check: results[check].get("output") for check in checks},
    }
