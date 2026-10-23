"""Compare the findings a legacy task raised with the ones its core check raised.

A legacy task raises a blurb per phase and column; the core check raises one finding
per subject. They agree when the strongest level of the blurbs equals the severity.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from tests.golden.migration.harness import Difference, Result

# The legacy guidance key for the dataset as a whole, and the core subject for it.
LEGACY_DATASET = "__dataset__"
_RANK = {"info": 0, "warn": 1, "error": 2}


def legacy_levels(guidance: Mapping[str, Any]) -> dict[str, str]:
    """The strongest level of the blurbs the task raised for each column."""
    levels: dict[str, str] = {}
    for column, phases in guidance.items():
        blurbs = [b for phase in phases.values() for b in phase]
        if blurbs:
            levels[column] = max((b["level"] for b in blurbs), key=_RANK.__getitem__)
    return levels


def core_levels(findings: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The strongest severity of the findings about each column.

    A finding about several columns counts for each of them, and one about the
    dataset for the legacy dataset key.
    """
    levels: dict[str, str] = {}
    for finding in findings:
        columns = finding["subject"]["columns"] or [LEGACY_DATASET]
        for column in columns:
            seen = levels.get(column)
            level = finding["severity"]
            if seen is None or _RANK[level] > _RANK[seen]:
                levels[column] = level
    return levels


def compare_levels(
    legacy: Mapping[str, Any],
    core: Mapping[str, Any],
    *,
    columns: Collection[str] | None = None,
    dataset_subject: str = "",
) -> list[Difference]:
    """Differences between the two sets of findings, optionally for some columns.

    ``columns`` limits the comparison to the columns the legacy task examined, for a
    task that shares its core check with another.
    """
    was = legacy_levels(legacy["result"].get("guidance") or {})
    now = core_levels(core["findings"])
    found: list[Difference] = []
    for column in sorted(set(was) | set(now)):
        if columns is not None and column not in columns and column != LEGACY_DATASET:
            continue
        before, after = was.get(column), now.get(column)
        subject = dataset_subject if column == LEGACY_DATASET else column
        if before is None:
            found.append(Difference(subject, "finding", f"core only ({after})"))
        elif after is None:
            found.append(Difference(subject, "finding", f"legacy only ({before})"))
        elif before != after:
            found.append(
                Difference(subject, "severity", f"legacy {before}, core {after}")
            )
    return found


def columns_read(old: Collection[str], new: Collection[str]) -> list[Difference]:
    """The columns only one of them read."""
    return [
        Difference(
            column, "column_read", "legacy only" if column in old else "core only"
        )
        for column in sorted(set(old) ^ set(new))
    ]


def only_codes(core: Result, codes: Collection[str]) -> Result:
    """The core result with just the findings of the given codes."""
    return {**core, "findings": [f for f in core["findings"] if f["code"] in codes]}


# Each legacy blurb title (a pattern) to the core finding code that carries it. A task
# raises an `eda` and an `ml` blurb for each case, and the core check raises one
# finding with the `ml` advice as tips on it, so both map to the same code. A blurb
# with no code would map to a reason it is dropped, as ``(pattern, None, reason)``.
LEGACY_FINDING_CODES: dict[str, tuple[tuple[str, str | None], ...]] = {
    "detect_zeros": (
        (r"^(High|Notable) Zero Rate \(", "high_zero_rate"),
        (r"^Zero-Inflated Column \(", "high_zero_rate"),
    ),
    "summarize_nulls": (
        (r"^(Severe|Significant|Some) Missingness \(", "high_null_rate"),
    ),
    "detect_constant_columns": ((r"^Constant Column - ", "constant_column"),),
    "detect_high_cardinality": ((r"^High Cardinality", "high_cardinality"),),
    "detect_id_columns": (
        (r"^Likely Identifier Column - ", "likely_identifier"),
        (r"^ID Column - ", "likely_identifier"),
    ),
    "detect_single_dominant_value": (
        (r"^Dominant Value ", "dominant_value"),
        (r"^Class Imbalance - ", "dominant_value"),
    ),
    "summarize_boolean_fields": (
        (r"^(Severe|Moderate) Imbalance \(", "dominant_value"),
        (r"^Class Imbalance \(", "dominant_value"),
    ),
    "detect_near_zero_variance": ((r"^Near-Zero Variance", "near_zero_variance"),),
    "detect_duplicates": ((r"^Duplicate Rows Detected \(", "duplicate_rows"),),
    "detect_duplicate_columns": (
        (r"^Exact Duplicate of ", "duplicate_columns"),
        (r"^Duplicate Column - ", "duplicate_columns"),
    ),
    "detect_skewness": ((r"^(Mild|Moderate|Heavy) (Left|Right) Skew \(", "skewed"),),
    "summarize_numeric": (
        (r"^Near-Zero Variance", "near_zero_variance"),
        (r"^(Mean-Median Gap|Distributional Asymmetry) \(", "mean_median_gap"),
    ),
    "compute_kurtosis": (
        (r"^(Fat Tails|Fat-Tailed Feature|Slightly Heavy Tails)", "heavy_tails"),
        (r"^Moderately Leptokurtic \(", "heavy_tails"),
        (r"^Thin Tails|^Thin-Tailed Feature", "thin_tails"),
    ),
    # The task raised no finding.
    "compute_entropy": (),
    "suggest_dtype_optimizations": ((r"^Dtype Optimization: ", "smaller_dtype"),),
    "sample_size_adequacy": (
        (r"^Insufficient Data for Linear", "too_few_rows_linear_model"),
        (r"^Insufficient Data for Tree", "too_few_rows_tree_model"),
        (r"^Insufficient Data for General", "too_few_rows_general_ml"),
    ),
    "detect_class_imbalance": (
        (r"^Class Imbalance - ", "class_imbalance"),
        (r"^Severe Class Imbalance", "class_imbalance"),
    ),
    # Its findings were reliability warnings, which are not guidance, and it is off by
    # default: the `schema` check raises findings of its own codes.
    "schema_validation": (),
    "detect_outliers": (
        (r"^Outliers: \d+ method\(s\) agree", "outliers_detected"),
        (r"^Outlier Sensitivity \(", "outliers_detected"),
        (r"^Multivariate Anomalies: ", "multivariate_anomalies"),
    ),
    "normality_tests": (
        (r"^Non-Normal Distribution \(", "non_normal"),
        (r"^Normality Assumption May Not Hold", "non_normal"),
    ),
    # The plot's pattern is part of the `non_normal` message and its Q-Q data.
    "normality_qq_plots": ((r"^QQ Plot: ", "non_normal"),),
    "detect_bimodal_distribution": (
        (r"^Bimodal Distribution \(", "bimodal"),
        (r"^Non-Gaussian Shape - Two Peaks", "bimodal"),
    ),
    "detect_out_of_bounds": ((r"^Out-of-Bounds Values", "out_of_bounds"),),
    # The task stored values and reliability warnings, and raised no finding.
    "compute_pairwise_associations": (),
    "contingency_tables": (
        (r"^Associated with ", "categorical_association"),
        (
            r"^(Strong|Moderate|Weak) Categorical Association with ",
            "categorical_association",
        ),
    ),
    "detect_data_leakage": (
        (r"^Near-Perfect Correlation with ", "redundant_columns"),
        (r"^Possible Data Leakage - Perfect Correlation with ", "redundant_columns"),
    ),
    "one_way_anova": ((r"^ANOVA: ", "group_difference"),),
    "kruskal_wallis": ((r"^Kruskal-Wallis: ", "group_difference"),),
    "mann_whitney_u": ((r"^Mann-Whitney: ", "group_difference"),),
    "detect_collinear_features": (
        (r"^(Moderate|High|Severe) Multicollinearity \(", "multicollinearity"),
        (r"^VIF = .* - Collinearity Impact on Models", "multicollinearity"),
    ),
    "missingness_mechanism_analysis": (
        (
            r"^Missingness Mechanism: consistent with (MAR|MCAR / MAR)",
            "missingness_depends_on_observed",
        ),
        (
            r"^Missingness Mechanism: consistent with (MCAR|Indeterminate)",
            "missingness_unexplained",
        ),
    ),
    "detect_mixed_type_columns": ((r"^Mixed Value Types \(", "mixed_value_types"),),
    "detect_encoded_columns": ((r"^Likely Encoded Column \(", "encoded_values"),),
    "detect_regex_format_violations": (
        (r"^Regex Format Violations \(", "format_violations"),
    ),
    "detect_string_anomalies": (
        (r"^Mixed Case - ", "mixed_case"),
        (r"^Whitespace Padding - ", "edge_whitespace"),
        (r"^Invisible Characters - ", "invisible_characters"),
        (r"^Length Outliers - ", "length_outliers"),
    ),
    "fuzzy_duplicate_detection": ((r"^Near-Duplicate Rows: ", "near_duplicate_rows"),),
}


def unmapped_titles(task: str, guidance: Mapping[str, Any]) -> list[str]:
    """Titles of the blurbs a task raised that no core finding code carries."""
    patterns = [
        re.compile(pattern) for pattern, _ in LEGACY_FINDING_CODES.get(task, ())
    ]
    return sorted(
        {
            blurb["title"]
            for phases in guidance.values()
            for blurbs in phases.values()
            for blurb in blurbs
            if not any(pattern.search(blurb["title"]) for pattern in patterns)
        }
    )
