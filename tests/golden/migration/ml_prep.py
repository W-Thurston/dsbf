"""The ML-prep guidance checks against the legacy tasks they replace.

The tasks stored a strategy per column as a string; the checks keep a typed strategy,
so each comparator maps the task's word to the check's before comparing.
"""

from __future__ import annotations

from tests.golden.migration.findings import columns_read, compare_levels
from tests.golden.migration.harness import Difference, Result

# The relative gap between the profile's bias-corrected skewness and the task's biased
# one is under 3% from 30 rows; the check also reads at most 20 000 rows.
_SKEW_TOLERANCE = 0.05

_ENCODINGS = {
    "one-hot": "one_hot",
    "frequency": "frequency",
    "frequency (high-cardinality)": "frequency",
}
_BINNINGS = {
    "equal-width binning": "equal_width",
    "quantile binning": "quantile",
    # The task's skew rule (a log transform above skewness 1) is the check's quantile
    # rule; the log transform itself is `transformations`' advice.
    "log-transform": "quantile",
}
# The task's strategy to the check's strategy and whether it adds an indicator;
# None means the strategy leaves the indicator to the column's tier.
_IMPUTATIONS = {
    "drop_or_indicator": ("indicator_and_drop", True),
    "mode": ("mode", False),
    "mode_with_indicator": ("mode", None),
    "forward_fill": ("forward_fill", None),
    "median_with_indicator": ("median", True),
    "median": ("median", False),
    "mean_or_knn_with_indicator": ("mean", True),
    "mean": ("mean", False),
    "mean_or_median": ("mean", False),
}


def compare_categorical_encoding(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]["encoding_suggestions"]
    new = core["output"]["columns"]
    found = columns_read(old, new)
    for column in sorted(set(old) & set(new)):
        was, now = old[column], new[column]
        if was["cardinality"] != now["cardinality"]:
            found.append(
                Difference(
                    column,
                    "cardinality",
                    f"legacy {was['cardinality']}, core {now['cardinality']}",
                )
            )
        text = was["suggested_encoding"]
        base = _ENCODINGS[text.replace(" + target encoding", "")]
        if base != now["strategy"]:
            found.append(
                Difference(column, "strategy", f"legacy {base}, core {now['strategy']}")
            )
        if "target encoding" in text:
            found.append(Difference(column, "target_encoding", "legacy only"))
    return found + compare_levels(legacy, core)


def compare_numeric_binning(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]["binning_suggestions"]
    new = core["output"]["columns"]
    found = columns_read(old, new)
    for column in sorted(set(old) & set(new)):
        was, now = _BINNINGS[old[column]["suggested_binning"]], new[column]["strategy"]
        if was != now:
            found.append(Difference(column, "strategy", f"legacy {was}, core {now}"))
    return found + compare_levels(legacy, core)


def compare_transformations(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]
    new = core["output"]["columns"]
    found = columns_read(old, new)
    for column in sorted(set(old) & set(new)):
        was = old[column]
        now = new[column]
        before = was["original_skewness"]
        if abs(before - now["skewness_before"]) > _SKEW_TOLERANCE * max(
            abs(before), 1e-3
        ):
            found.append(
                Difference(
                    column,
                    "skewness",
                    f"legacy {before:.4f}, core {now['skewness_before']:.4f}",
                )
            )
        chosen = next(
            (n for n, t in was["transforms"].items() if t.get("recommended")), None
        )
        if chosen != now["recommended"]:
            found.append(
                Difference(
                    column,
                    "recommended",
                    f"legacy {chosen}, core {now['recommended']}",
                )
            )
    return found + compare_levels(legacy, core)


def compare_imputation(legacy: Result, core: Result) -> list[Difference]:
    old = legacy["result"]["data"]["suggestions"]
    new = core["output"]["columns"]
    found = columns_read(old, new)
    for column in sorted(set(old) & set(new)):
        was, now = old[column], new[column]
        if was["tier"] != now["tier"]:
            found.append(
                Difference(column, "tier", f"legacy {was['tier']}, core {now['tier']}")
            )
        strategy, indicator = _IMPUTATIONS[was["strategy"]]
        if strategy != now["strategy"]:
            found.append(
                Difference(
                    column,
                    "strategy",
                    f"legacy {was['strategy']}, core {now['strategy']}",
                )
            )
        wanted = was["add_indicator"] if indicator is None else indicator
        if wanted != now["add_indicator"]:
            found.append(
                Difference(
                    column,
                    "indicator",
                    f"legacy {wanted}, core {now['add_indicator']}",
                )
            )
    return found + compare_levels(legacy, core)
