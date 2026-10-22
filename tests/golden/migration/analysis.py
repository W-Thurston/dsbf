"""The distribution analysis checks against the legacy tasks they replace."""

from __future__ import annotations

import math

from tests.golden.migration.findings import (
    LEGACY_DATASET,
    columns_read,
    compare_levels,
    only_codes,
)
from tests.golden.migration.harness import Difference, Result

_RELATIVE = 1e-5  # the tasks rounded to six decimals
_ABSOLUTE = 1e-9
# A p-value this small is zero for every decision, and the two scipy calls differ in
# its digits.
_TINY_P = 1e-12

_TEST_NAMES = {"shapiro_wilk": "shapiro_wilk", "ks_normal": "ks_fitted_normal"}


def _close(was: float, now: float) -> bool:
    return math.isclose(was, now, rel_tol=_RELATIVE, abs_tol=_ABSOLUTE)


def _column_findings(legacy: Result) -> Result:
    """The legacy result without its dataset-level guidance."""
    guidance = legacy["result"].get("guidance") or {}
    rest = {k: v for k, v in guidance.items() if k != LEGACY_DATASET}
    return {**legacy, "result": {**legacy["result"], "guidance": rest}}


def compare_outliers(legacy: Result, core: Result) -> list[Difference]:
    """Per column: the count each rule flags and the fences; then the forest."""
    old = {k: v for k, v in legacy["result"]["data"].items() if k != LEGACY_DATASET}
    new = core["output"]["outliers"]["columns"]
    found = columns_read(old, new)
    for column in sorted(set(old) & set(new)):
        was, now = old[column], new[column]
        for method in ("iqr", "zscore", "mad"):
            before, after = was.get(method), now[method]
            if (before is None) != (after is None):
                found.append(Difference(column, method, "legacy and core disagree"))
            elif before is not None:
                if before["outlier_count"] != after["count"]:
                    found.append(
                        Difference(
                            column,
                            f"{method}_count",
                            f"legacy {before['outlier_count']}, core {after['count']}",
                        )
                    )
        if was["consensus"] != now["consensus"]:
            found.append(
                Difference(
                    column,
                    "consensus",
                    f"legacy {was['consensus']}, core {now['consensus']}",
                )
            )
        fences = (
            (now["iqr"] or {}).get("lower_fence"),
            (now["iqr"] or {}).get("upper_fence"),
        )
        for key, value in zip(("lower_fence", "upper_fence"), fences, strict=True):
            if value is not None and not _close(was["iqr"][key], value):
                found.append(
                    Difference(
                        column, key, f"legacy {was['iqr'][key]}, core {value:.6g}"
                    )
                )
    found += compare_levels(
        _column_findings(legacy), only_codes(core, {"outliers_detected"})
    )
    return found + _compare_forest(legacy, core)


def _compare_forest(legacy: Result, core: Result) -> list[Difference]:
    was = legacy["result"]["data"].get(LEGACY_DATASET, {}).get("isolation_forest")
    now = core["output"]["multivariate_outliers"]
    found: list[Difference] = []
    if core["status"] != "ok" or now is None:
        return found
    if was is None:
        return [Difference("", "isolation_forest", "core only")]
    if sorted(was["feature_columns"]) != sorted(now["feature_columns"]):
        found.append(
            Difference(
                "",
                "forest_features",
                f"legacy {len(was['feature_columns'])}, "
                f"core {len(now['feature_columns'])}",
            )
        )
    if was["n_flagged"] != now["n_flagged"]:
        found.append(
            Difference(
                "",
                "forest_flagged",
                f"legacy {was['n_flagged']}, core {now['n_flagged']}",
            )
        )
    return found


def compare_normality(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]
    new = core["output"]["columns"]
    found = columns_read(old, new)
    for column in sorted(set(old) & set(new)):
        was, now = old[column], new[column]
        test = _TEST_NAMES[was["primary_test"]["test"]]
        if test != now["primary"]["test"]:
            found.append(
                Difference(
                    column,
                    "primary_test",
                    f"legacy {test}, core {now['primary']['test']}",
                )
            )
            continue
        pairs = (
            ("primary", was["primary_test"], now["primary"]),
            ("jarque_bera", was["jarque_bera"], now["jarque_bera"]),
        )
        for label, before, after in pairs:
            if not _close(before["statistic"], after["statistic"]):
                found.append(
                    Difference(
                        column,
                        f"{label}_statistic",
                        f"legacy {before['statistic']:.6g}, "
                        f"core {after['statistic']:.6g}",
                    )
                )
            tiny = before["p_value"] < _TINY_P and after["p_value"] < _TINY_P
            if not tiny and not _close(before["p_value"], after["p_value"]):
                found.append(
                    Difference(
                        column,
                        f"{label}_p_value",
                        f"legacy {before['p_value']:.6g}, core {after['p_value']:.6g}",
                    )
                )
        verdict = "non_normal" if now["non_normal"] else "normal"
        if was["overall_verdict"] != verdict:
            found.append(
                Difference(
                    column,
                    "verdict",
                    f"legacy {was['overall_verdict']}, core {verdict}",
                )
            )
    return found + compare_levels(legacy, core)


def compare_qq(legacy: Result, core: Result) -> list[Difference]:
    """The quantile pairs, for the columns the task computed them for.

    The task's findings describe a plot, and carry the level of the normality verdict;
    the core finding is `normality`'s, so only the data is compared here.
    """
    old = legacy["result"]["data"]
    new = {c: v["qq"] for c, v in core["output"]["columns"].items() if v["qq"]}
    found = columns_read(old, new)
    for column in sorted(set(old) & set(new)):
        was, now = old[column], new[column]
        pairs = (
            ("theoretical", was["theoretical"], now["theoretical"]),
            ("empirical", was["empirical"], now["empirical"]),
        )
        for label, before, after in pairs:
            if len(before) != len(after) or not all(
                _close(a, b) for a, b in zip(before, after, strict=True)
            ):
                found.append(Difference(column, f"qq_{label}", "the quantiles differ"))
    return found


def compare_bimodality(legacy: Result, core: Result) -> list[Difference]:
    scores = legacy["result"]["data"]["bic_scores"]
    flags = legacy["result"]["data"]["bimodal_flags"]
    new = core["output"]["columns"]
    fitted = {c for c, s in scores.items() if s.get("bic_1_component") is not None}
    found = columns_read(fitted, new)
    for column in sorted(fitted & set(new)):
        was, now = scores[column], new[column]
        if now["n_used"] == now["n_finite"]:
            for key, label in (
                ("bic_1_component", "bic_1_component"),
                ("bic_2_components", "bic_2_components"),
                ("delta", "bic_delta"),
            ):
                if not _close(was[key], now[label]):
                    found.append(
                        Difference(
                            column,
                            key,
                            f"legacy {was[key]:.6g}, core {now[label]:.6g}",
                        )
                    )
        if flags[column] != now["bimodal"]:
            found.append(
                Difference(
                    column, "bimodal", f"legacy {flags[column]}, core {now['bimodal']}"
                )
            )
    return found + compare_levels(legacy, only_codes(core, {"bimodal"}))
