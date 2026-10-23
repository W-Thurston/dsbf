"""What differs between the datetime and time-series checks and their tasks.

See ``accepted.py`` for what an entry is. The rows are what ``test_migration_parity``
printed for the golden datasets, grouped by the reason each is right. Only kitchen5k
has a time index (``normal_date``, daily with 14 rows a day); the tasks tested its 5 250
rows in file order, the checks its 365 daily means.
"""

from __future__ import annotations

from collections.abc import Iterable

from tests.golden.migration.harness import Accepted

NO_INDEX = (
    "No time index is set on this dataset, so the task, with `time_series.enabled` "
    "false, ended ok with a message that it was disabled and no result. The check "
    "ends not applicable (`no_time_index`) with a hint to pass `time_index=...`."
)
CATEGORICAL = (
    "The profile types these columns categorical (two to eleven values: targets, "
    "codes and flags), so the check, which applies to numeric columns, leaves them "
    "out. The task analysed every pandas numeric column, and a unit root or an "
    "autocorrelation of a 0/1 target means nothing."
)
LAG_BAND = (
    "The task listed every one of 40 lags outside a 95% band on 5 250 rows in file "
    "order, so a column of independent draws has a lag outside it with probability "
    "1 - 0.95^40 = 87%, and the band (±0.027) is narrower than any effect worth "
    "reading. The check uses the 365 daily means, whose band is ±0.10, and flags a "
    "lag only when it is also at least 0.2; this column's daily means show none."
)
NO_PERIOD = (
    "The task takes its seasonal period from the frequency it infers from the "
    "rows, found none on a daily index that repeats each date 14 times, and "
    "decomposed nothing. The check resamples to the profile's frequency (`D`), "
    "takes the period of 7 from it and decomposes the 22 numeric columns."
)
KPSS_AT_ALPHA = (
    "A borderline KPSS: the p-value is next to α = 0.05 (legacy 0.048, core 0.1 or "
    "above on the 365 daily means), so the reading flips between two samples of "
    "the same column. The ADF result is a clear rejection in both."
)
KPSS_AT_ALPHA_OTHER_WAY = (
    "A borderline KPSS: the task's p-value was 0.073 on the 5 250 rows and the "
    "check's 0.025 on the 365 daily means, either side of α = 0.05, while both ADF "
    "p-values are about 0. The check reports the disagreement as information "
    "(`ambiguous_stationarity`) and does not call the column non-stationary."
)
RAMP = (
    "A deterministic ramp. On the 365 daily means ADF with a constant has too little "
    "power against a linear trend to reject a unit root (p = 0.073) and KPSS rejects "
    "(p = 0.01), so the check reports a drifting level and its tip says to difference "
    "or detrend; the task's ADF rejected (p = 0) with 18 lags on 5 250 rows."
)
UNPARSED = (
    "The task counted what pandas could not turn into a nanosecond timestamp: the 104 "
    "missing values, which `null_rate` reports, and 553 dates after 2262-04-11, "
    "pandas' last. The values are valid dates written the same way as the rest (the "
    "column holds planted values up to 2430), so the check, which reads them with the "
    "format the profile found, counts none unparsed; the out-of-range ones are "
    "`outliers` and `out_of_bounds` matters."
)
IRREGULAR = (
    "The dates are scattered over 407 years (2023 to 2430), so no step covers half "
    "of the 4 638 distinct values and the column is not a regular series. The task "
    "counted 3 939 intervals over twice the mode of the rounded day gaps; the check "
    "says there is no regular spacing (`irregular_spacing`, the longest step seven "
    "years) and, being information, is not the task's warning."
)
MONTH_FEATURES = (
    "`signup` runs through one month, so no month feature varies. The task listed "
    "month, quarter and week of year because its test of which parts vary was always "
    "true."
)


def _group(
    task: str, dataset: str, reason: str, rows: Iterable[tuple[str, str, str]]
) -> list[Accepted]:
    return [Accepted(task, dataset, s, k, d, reason) for s, k, d in rows]


def _legacy_only(columns: Iterable[str]) -> list[tuple[str, str, str]]:
    return [(column, "column_read", "legacy only") for column in columns]


_CATEGORICAL_COLUMNS = [
    "balanced_target",
    "constant_numeric",
    "imbalanced_target",
    "mostly_zeros",
    "multiclass_target",
    "near_zero_variance",
    "perfect_leakage",
    "target_for_leakage",
]
_BAND_COLUMNS = [
    "bimodal_column",
    "corr_base",
    "duplicate_col_1",
    "duplicate_col_2",
    "duplicate_col_3",
    "extreme_skew",
    "highly_correlated",
    "moderately_correlated",
    "negatively_correlated",
    "normal_numeric_float",
    "normal_numeric_int",
    "outlier_column",
    "partial_duplicate_1",
    "partial_duplicate_2",
    "percentage_out_of_bounds",
    "right_skewed",
    "sparse_column",
    "subtle_leakage",
]
_NO_INDEX_DATASETS = (
    "clean",
    "tiny",
    "near_clean",
    "high_missingness",
    "severe_multicollinearity",
    "single_column",
    "wide",
    "titanic",
)
_TASKS_NEEDING_AN_INDEX = (
    "detect_stationarity",
    "compute_acf_pacf",
    "decompose_time_series",
)

ACCEPTED_TIME_SERIES: tuple[Accepted, ...] = (
    *(
        accepted
        for task in _TASKS_NEEDING_AN_INDEX
        for dataset in _NO_INDEX_DATASETS
        for accepted in _group(
            task, dataset, NO_INDEX, [("", "status", "legacy ok, core not_applicable")]
        )
    ),
    *_group(
        "extract_datetime_features",
        "tiny",
        MONTH_FEATURES,
        [
            (
                "signup",
                "features",
                "legacy only ['month', 'month_sin_cos', 'quarter', 'week_of_year'], "
                "core only []",
            )
        ],
    ),
    *_group(
        "check_datetime_consistency",
        "kitchen5k",
        UNPARSED,
        [("datetime_with_gaps", "unparsed", "legacy 657, core 0")],
    ),
    *_group(
        "temporal_gap_detection",
        "kitchen5k",
        IRREGULAR,
        [
            ("datetime_with_gaps", "gaps", "legacy 3939, core irregular"),
            ("datetime_with_gaps", "severity", "legacy warn, core info"),
        ],
    ),
    *_group(
        "detect_stationarity",
        "kitchen5k",
        CATEGORICAL,
        _legacy_only(_CATEGORICAL_COLUMNS),
    ),
    *_group(
        "compute_acf_pacf",
        "kitchen5k",
        CATEGORICAL,
        _legacy_only(_CATEGORICAL_COLUMNS),
    ),
    *_group(
        "compute_acf_pacf",
        "kitchen5k",
        LAG_BAND,
        [(column, "autocorrelated", "legacy yes, core no") for column in _BAND_COLUMNS],
    ),
    *_group(
        "detect_stationarity",
        "kitchen5k",
        KPSS_AT_ALPHA,
        [
            (
                "moderately_correlated",
                "assessment",
                "legacy ambiguous_trend_stationary_or_structural_break, "
                "core consistent_with_stationary",
            )
        ],
    ),
    *_group(
        "detect_stationarity",
        "kitchen5k",
        KPSS_AT_ALPHA_OTHER_WAY,
        [
            (
                "subtle_leakage",
                "assessment",
                "legacy consistent_with_stationary, "
                "core ambiguous_trend_stationary_or_structural_break",
            )
        ],
    ),
    *_group(
        "detect_stationarity",
        "kitchen5k",
        RAMP,
        [
            (
                "sequential_id",
                "assessment",
                "legacy ambiguous_trend_stationary_or_structural_break, "
                "core consistent_with_non_stationary",
            )
        ],
    ),
    *_group(
        "decompose_time_series",
        "kitchen5k",
        NO_PERIOD,
        [("", "columns", "legacy 0, core 22")],
    ),
)
