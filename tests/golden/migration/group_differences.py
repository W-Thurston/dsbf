"""The group difference and collinearity checks against the tasks they replace."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tests.golden.migration.findings import compare_levels
from tests.golden.migration.harness import Difference, Result

_F_ABSOLUTE = 6e-5  # the tasks rounded F and H to four decimals
_P_ABSOLUTE = 2e-6  # and p to six
_R_ABSOLUTE = 2e-4
_VIF_RELATIVE = 1e-2
ALPHA = 0.05


def _moved(subject: str, kind: str, was: float, now: float) -> Difference:
    return Difference(subject, kind, f"legacy {was:.6g}, core {now:.6g}")


def _legacy_pairs(data: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """The legacy entries by (numeric, categorical)."""
    pairs: dict[tuple[str, str], Mapping[str, Any]] = {}
    for key, entry in data.items():
        if key.startswith("__"):
            continue
        numeric, categorical = key.split("|", 1)
        pairs[(numeric, categorical)] = entry
    return pairs


def _core_pairs(core: Result) -> dict[tuple[str, str], Mapping[str, Any]]:
    # A check with nothing to read ended not applicable and has no output.
    output = core["output"] or {}
    return {(p["numeric"], p["categorical"]): p for p in output.get("pairs", ())}


def _by_method(core: Result, method: str) -> Result:
    """The core result with just the findings that a method flagged."""
    return {
        **core,
        "findings": [
            f for f in core["findings"] if method in f["evidence"]["method"].split("+")
        ],
    }


def _omnibus(
    legacy: Result, core: Result, *, statistic: str, p_value: str, adjusted: str
) -> list[Difference]:
    old = _legacy_pairs(legacy["result"]["data"])
    new = _core_pairs(core)
    found: list[Difference] = []
    for pair in sorted(set(old) | set(new)):
        subject = "|".join(pair)
        if pair not in new:
            found.append(Difference(subject, "pair", "legacy only"))
            continue
        if pair not in old:
            found.append(Difference(subject, "pair", "core only"))
            continue
        was, now = old[pair], new[pair]
        legacy_stat = was["f_statistic" if statistic == "f" else "h_statistic"]
        core_stat = now["f_statistic" if statistic == "f" else "h_statistic"]
        if core_stat is None or not math.isclose(
            legacy_stat, core_stat, abs_tol=_F_ABSOLUTE, rel_tol=1e-6
        ):
            found.append(_moved(subject, statistic, legacy_stat, core_stat or math.nan))
        if not math.isclose(was["p_value"], now[p_value], abs_tol=_P_ABSOLUTE):
            found.append(_moved(subject, "p_value", was["p_value"], now[p_value]))
        if was["n_total"] != now["n_used"]:
            found.append(_moved(subject, "n", was["n_total"], now["n_used"]))
        if was["n_groups"] != now["n_groups"]:
            found.append(_moved(subject, "n_groups", was["n_groups"], now["n_groups"]))
        if was["significant"] != (now[adjusted] < ALPHA):
            found.append(
                Difference(
                    subject,
                    "significance",
                    f"legacy {was['significant']}, core {now[adjusted] < ALPHA}",
                )
            )
    return found


def compare_one_way_anova(legacy: Result, core: Result) -> list[Difference]:
    """F, p, the sample and the significance of every pair, then the findings."""
    return _omnibus(
        legacy,
        core,
        statistic="f",
        p_value="anova_p",
        adjusted="anova_p_adjusted",
    ) + compare_levels(legacy, _by_method(core, "anova"))


def compare_kruskal_wallis(legacy: Result, core: Result) -> list[Difference]:
    """H, p, the sample and the significance of every pair, then the findings."""
    return _omnibus(
        legacy,
        core,
        statistic="h",
        p_value="kruskal_p",
        adjusted="kruskal_p_adjusted",
    ) + compare_levels(legacy, _by_method(core, "kruskal"))


def compare_mann_whitney_u(legacy: Result, core: Result) -> list[Difference]:
    """The contrasts both ran, and a count of the ones only one side ran.

    The task tested the first ten level pairs of every pair of columns, and the check
    contrasts every level pair of the pairs it flags, so most tests are on one side
    only. The counts are one difference each, so the numbers they cover stay visible.
    """
    old = {
        (e["num_col"], e["cat_col"], frozenset((e["level_a"], e["level_b"]))): e
        for key, e in legacy["result"]["data"].items()
        if not key.startswith("__")
    }
    new: dict[tuple[str, str, frozenset[str]], Mapping[str, Any]] = {}
    for pair in _core_pairs(core).values():
        for contrast in pair["contrasts"]:
            levels = frozenset((contrast["first_level"], contrast["second_level"]))
            new[(pair["numeric"], pair["categorical"], levels)] = contrast
    found: list[Difference] = []
    for key in sorted(set(old) & set(new), key=lambda k: (k[0], k[1], sorted(k[2]))):
        was, now = old[key], new[key]
        subject = f"{key[0]}|{key[1]}|{was['level_a']}_vs_{was['level_b']}"
        # The task's sign took U of the first group, so it is the opposite of the
        # check's, whose positive value means the first level is higher.
        expected = (
            -was["rank_biserial_r"]
            if now["first_level"] == was["level_a"]
            else was["rank_biserial_r"]
        )
        if not math.isclose(expected, now["rank_biserial"], abs_tol=_R_ABSOLUTE):
            found.append(
                _moved(subject, "rank_biserial", expected, now["rank_biserial"])
            )
        if not math.isclose(was["p_value"], now["p_value"], abs_tol=_P_ABSOLUTE):
            found.append(_moved(subject, "p_value", was["p_value"], now["p_value"]))
    only_old = len(set(old) - set(new))
    only_new = len(set(new) - set(old))
    if only_old:
        found.append(Difference("", "contrasts", f"{only_old} tested by the task only"))
    if only_new:
        found.append(Difference("", "contrasts", f"{only_new} run by the check only"))
    return found + compare_levels(legacy, core)


def compare_collinear_features(legacy: Result, core: Result) -> list[Difference]:
    """The VIF of every column, then the findings.

    A column is a difference when its VIF differs by more than 1%; the detail shows
    three significant figures, since the two definitions differ in more than rounding.
    """
    was = legacy["result"]["data"]["vif_scores"]
    output = core["output"] or {}
    now = {c["column"]: c["vif"] for c in output.get("columns", ())}
    found: list[Difference] = []
    for column in sorted(set(was) | set(now)):
        if column not in now:
            found.append(Difference(column, "vif", "legacy only"))
        elif column not in was:
            found.append(Difference(column, "vif", "core only"))
        elif not math.isclose(was[column], now[column], rel_tol=_VIF_RELATIVE):
            found.append(
                Difference(
                    column, "vif", f"legacy {was[column]:.3g}, core {now[column]:.3g}"
                )
            )
    return found + compare_levels(legacy, core)
