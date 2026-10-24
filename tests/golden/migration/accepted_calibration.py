"""The kinds of difference accepted for the checks calibrated against planted truth.

See ``harness.AcceptedKind`` for why these are accepted as kinds. Each entry says what
the calibration changed; the rates before and after are in `PLAN.md`'s history and in
`tests/calibration/baseline.json`.
"""

from __future__ import annotations

from tests.golden.migration.harness import AcceptedKind

ACCEPTED_CALIBRATION: tuple[AcceptedKind, ...] = (
    AcceptedKind(
        "one_way_anova",
        ("p_value", "f", "significance"),
        "`group_differences` takes Welch's ANOVA, which does not assume the groups "
        "have one spread. With groups of unequal size and spread the classic F that "
        "the task used called equal means different in 55% of runs, and Welch's in "
        "6% (nominal 5%).",
    ),
    AcceptedKind(
        "kruskal_wallis",
        ("p_value",),
        "`group_differences` takes Welch's ANOVA on the ranks: Kruskal-Wallis, "
        "which the task used, called equal groups different in 20% of runs when the "
        "groups differ in size and spread, and Welch on ranks in 6%.",
    ),
    AcceptedKind(
        "detect_outliers",
        (
            "upper_fence",
            "lower_fence",
            "iqr_count",
            "zscore_count",
            "mad_count",
            "finding",
            "consensus",
            "status",
            "severity",
            "forest_features",
        ),
        "`outliers` raises its cutoffs with the number of values until a normal "
        "column is flagged by chance 1% of the time (the fixed 1.5 IQR and 3 "
        "deviations flagged 83% of clean normal columns of 3 000 values), and "
        "`multivariate_outliers` calls a row far only beyond a chi-squared limit of "
        "a robust distance, or when the forest's top score stands above a normal "
        "sample's, and leaves out a column that is mostly missing.",
    ),
    AcceptedKind(
        "normality_tests",
        ("verdict", "finding"),
        "`normality` holds each of its two tests to half the level and reports a "
        "rejection only when the quantiles are 0.05 deviations from a normal's on "
        "average: with both tests at the full level 17% of normal columns were "
        "rejected, and a t distribution with 30 degrees of freedom is rejected at "
        "50 000 rows with a gap of 0.02.",
    ),
    AcceptedKind(
        "normality_qq_plots",
        ("column_read",),
        "The Q-Q data of a column is kept only for the columns `normality` reports, "
        "which fewer now are.",
    ),
    AcceptedKind(
        "detect_feature_drift",
        ("flagged", "level"),
        "`reference_drift` subtracts the PSI two samples of one population reach by "
        "chance and needs a chi-squared test under 0.01 as well, so a small sample "
        "is not drifting (93% of 150-row pairs of one population were).",
    ),
    AcceptedKind(
        "detect_target_drift",
        ("level",),
        "`reference_drift` holds the target to half the limits of a feature: a "
        "15-point move in the share of positives was not found.",
    ),
    AcceptedKind(
        "data_quality_scorer",
        ("level", "core_only"),
        "`quality_dimensions` follows a written rule: a warning is amber and red "
        "above half the columns, a finding below a warning is blue up to 30% of the "
        "columns and amber above. The checks it reads are calibrated too, so the "
        "columns a dimension flags move with them.",
    ),
    AcceptedKind(
        "ml_readiness_scorer",
        ("level", "legacy_only"),
        "`ml_readiness` reads the findings of the calibrated checks, so the columns "
        "with an action move with them.",
    ),
    AcceptedKind(
        "detect_string_anomalies",
        ("length_outliers", "finding"),
        "`string_anomalies` gives a length fence at least one character of spread: "
        "a column whose values share a length had none, so a value far from it was "
        "never an outlier.",
    ),
    AcceptedKind(
        "transformation_preview",
        ("recommended",),
        "`transformations` suggests a fallback transform only if it leaves the "
        "column under the skew threshold: the fitted power of a column with a spike "
        "at one value moves with the number of rows, and the suggestion with it.",
    ),
    AcceptedKind(
        "summarize_numeric",
        ("severity", "finding"),
        "`skewness` reports a mean-median gap from 0.35 deviations and warns from "
        "0.45: the warn level the task used (1.0) cannot be reached, since the mean "
        "is at most one deviation from the median.",
    ),
    AcceptedKind(
        "fuzzy_duplicate_detection",
        ("pairs_flagged",),
        "`fuzzy_duplicates` caps two values that are the same but for their "
        "numbers at 0.7 alike, so records that differ in a serial or a street "
        "number are not pairs.",
    ),
    AcceptedKind(
        "detect_stationarity",
        ("assessment",),
        "`stationarity` also tests with a trend: a series both constant-only tests "
        "call drifting or ambiguous but that is stationary around a trend is read "
        "as that.",
    ),
)
