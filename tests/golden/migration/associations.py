"""The association and contingency checks against the legacy tasks they replace."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tests.golden.migration.findings import compare_levels, only_codes
from tests.golden.migration.harness import Difference, Result

_ROUNDED = 2e-6  # the tasks rounded to six decimals
_CHI2_RELATIVE = 1e-4  # contingency_tables rounded chi-squared to four decimals
_P_ABSOLUTE = 2e-6
_V_ABSOLUTE = 1e-4  # and V to four

Pair = frozenset[str]


def _key(first: str, second: str) -> Pair:
    return frozenset((first, second))


def _label(pair: Pair, order: Mapping[Pair, str]) -> str:
    return order.get(pair) or "|".join(sorted(pair))


def _moved(subject: str, kind: str, was: float, now: float) -> Difference:
    return Difference(subject, kind, f"legacy {was:.6g}, core {now:.6g}")


def _legacy_pairs(data: Mapping[str, Any]) -> dict[Pair, tuple[str, Any]]:
    """The legacy entries by unordered pair, with the key the task wrote."""
    pairs: dict[Pair, tuple[str, Any]] = {}
    for key, entry in data.items():
        if key.startswith("__"):
            continue
        first, second = key.split("|", 1)
        pairs[_key(first, second)] = (key, entry)
    return pairs


def _close(was: float, now: float, absolute: float = _ROUNDED) -> bool:
    return math.isclose(was, now, abs_tol=absolute, rel_tol=0)


def compare_pairwise_associations(legacy: Result, core: Result) -> list[Difference]:
    """The value of every pair either one computed, on the one scale they share.

    The task raised no finding (it stored values and reliability warnings), so
    only the values are compared. Point-biserial r and η² are compared with the
    absolute value and the square of the check's η, and V with its plain V.
    """
    old = _legacy_pairs(legacy["result"]["data"])
    new: dict[Pair, tuple[str, Mapping[str, Any]]] = {}
    # A check with nothing to read ended not applicable and has no output.
    for source in ("associations", "contingency"):
        for pair in (core["output"][source] or {}).get("pairs", ()):
            new[_key(pair["first"], pair["second"])] = (source, pair)
    labels = {pair: key for pair, (key, _) in old.items()}
    found: list[Difference] = []
    for pair in sorted(set(old) | set(new), key=sorted):
        subject = _label(pair, labels)
        if pair not in new:
            found.append(Difference(subject, "pair", "legacy only"))
            continue
        if pair not in old:
            found.append(Difference(subject, "pair", "core only"))
            continue
        _, was = old[pair]
        source, now = new[pair]
        found += _compare_value(subject, was, source, now)
    return found


# The core metric that stands for each legacy one, and how to put the core value on
# the legacy one's scale: point-biserial r is signed and η is not, and η² is η squared.
_COUNTERPART = {
    "pearson_r": ("pearson_r", lambda value: value),
    "spearman_r": ("spearman_r", lambda value: value),
    "kendalls_tau": ("kendalls_tau", lambda value: value),
    "point_biserial_r": ("eta", lambda value: value),
    "eta_squared": ("eta", lambda value: value**2),
    "cramers_v": ("cramers_v", lambda value: value),
}


def _compare_value(
    subject: str, was: Mapping[str, Any], source: str, now: Mapping[str, Any]
) -> list[Difference]:
    metric, value = was["metric_type"], was["metric"]
    expected, scale = _COUNTERPART[metric]
    if source == "contingency":
        got, core_value = "cramers_v", now["cramers_v"]
    else:
        got, core_value = now["metric"], now["value"]
    if got != expected:
        return [Difference(subject, "metric", f"legacy {metric}, core {got}")]
    core_value = scale(core_value)
    if metric == "point_biserial_r":
        value = abs(value)
    tolerance = _V_ABSOLUTE if metric == "cramers_v" else _ROUNDED
    if _close(value, core_value, tolerance):
        return []
    return [_moved(subject, metric, value, core_value)]


def compare_data_leakage(legacy: Result, core: Result) -> list[Difference]:
    """The pairs with |r| at the threshold, against the redundancy findings.

    The task called a near-copy an error and "leakage"; `redundant_columns` is a
    warning about redundancy and says it is no statement about a target, so the
    levels are not compared.
    """
    old = _legacy_pairs(legacy["result"]["data"]["leakage_pairs"])
    new = {
        _key(*f["subject"]["columns"]): f
        for f in only_codes(core, {"redundant_columns"})["findings"]
    }
    labels = {pair: key for pair, (key, _) in old.items()}
    found: list[Difference] = []
    for pair in sorted(set(old) | set(new), key=sorted):
        subject = _label(pair, labels)
        if pair not in new:
            found.append(Difference(subject, "pair", "legacy only"))
        elif pair not in old:
            found.append(Difference(subject, "pair", "core only"))
        else:
            was = old[pair][1]
            now = abs(new[pair]["evidence"]["metrics"]["pearson_r"])
            if not _close(was, now, 1e-4):
                found.append(
                    Difference(
                        subject, "correlation", f"legacy {was:.6g}, core {now:.6g}"
                    )
                )
    return found


def compare_contingency_tables(legacy: Result, core: Result) -> list[Difference]:
    """Per pair: the table's size, the test and the plain V; then the findings.

    The task cut a table to its top levels before testing it, so a truncated table
    is reported once and its statistics are not compared.
    """
    old = _legacy_pairs(legacy["result"]["data"])
    new = {_key(p["first"], p["second"]): p for p in core["output"]["pairs"]}
    labels = {pair: key for pair, (key, _) in old.items()}
    found: list[Difference] = []
    for pair in sorted(set(old) | set(new), key=sorted):
        subject = _label(pair, labels)
        if pair not in new:
            found.append(Difference(subject, "pair", "legacy only"))
            continue
        if pair not in old:
            found.append(Difference(subject, "pair", "core only"))
            continue
        was, now = old[pair][1], new[pair]
        if was["truncated"]:
            found.append(
                Difference(subject, "truncated_table", "legacy tested the top levels")
            )
            continue
        if was["n"] != now["n_used"]:
            found.append(_moved(subject, "n", was["n"], now["n_used"]))
        if was["dof"] != now["dof"]:
            found.append(_moved(subject, "dof", was["dof"], now["dof"]))
        chi2 = (was["chi2"], now["chi2"])
        if not math.isclose(*chi2, rel_tol=_CHI2_RELATIVE, abs_tol=1e-4):
            found.append(_moved(subject, "chi2", *chi2))
        p_values = (was["p_value"], now["p_value"])
        if not _close(*p_values, _P_ABSOLUTE):
            found.append(_moved(subject, "p_value", *p_values))
        v = (was["cramers_v"], now["cramers_v"])
        if not _close(*v, _V_ABSOLUTE):
            found.append(_moved(subject, "cramers_v", *v))
    return found + compare_levels(legacy, only_codes(core, {"categorical_association"}))
