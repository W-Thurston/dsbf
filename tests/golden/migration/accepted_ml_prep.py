"""What differs between the ML-prep checks and the tasks they replace.

See ``accepted.py`` for what an entry is. The rows are what ``test_migration_parity``
printed for the golden datasets, grouped by the reason each is right.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tests.golden.migration.harness import Accepted

NOT_CATEGORICAL = (
    "The task read every column of string dtype, whatever it holds. The profile types "
    "free text, identifiers (UUIDs, user and transaction ids), dates stored as text, "
    "encoded strings, mixed-type columns and constant or all-null columns as `text`, "
    "`identifier`, `datetime` or `unknown`, and `categorical_encoding` reads only "
    "categorical columns, so it doesn't advise one-hot or frequency encoding for a "
    "column that is not a category. Free text and identifiers have their own checks."
)
CODE_COLUMN = (
    "The profile types an integer column categorical when it has two values, or few "
    "distinct values under 5% of the rows: a label or a small code (class, number of "
    "siblings, the 0/1 targets and the columns planted to leak them). The task read "
    "string dtypes only, so it never advised on them; encoding is the right question "
    "for a label, so the check does."
)
BINARY = (
    "A column of two values needs one 0/1 column. Two one-hot columns carry the same "
    "information, and being each other's reverse they are collinear in a linear "
    "model. The task suggested one-hot for any cardinality up to 10, two included."
)
SPARSE = (
    "`category` has 25 rows in tiny, too few per value for a one-hot column to be "
    "estimated (the check wants 20 rows per value), so it suggests the frequency of "
    "each value. The task looked at the number of values only."
)
ADVICE_SEVERITY = (
    "Choosing an encoding or a binning is advice, not a defect, so the check reports "
    "it at `info`. The task raised `warn` for a column of more than 50 values "
    "(`high_cardinality` reports the column itself) and for a skewed column it "
    "advised to log-transform (`skewness` reports the skew and `transformations` the "
    "transform)."
)
CODE_COLUMN_BINNING = (
    "The profile types an integer column with few distinct values categorical (a "
    "label or a code; `categorical_encoding` advises on it), so `numeric_binning` "
    "doesn't read it. The task binned every numeric dtype, including the 0/1 targets "
    "and class codes."
)
TOO_FEW_ROWS = (
    "tiny has 25 rows. A suggestion needs two bins of 30 rows (`min_rows_per_bin`), "
    "so the check skips the column and says why in `skipped_columns`. The task "
    "suggested a strategy for 25 values."
)
LISTWISE_SKEW = (
    "The task took the skewness from `compute_reliability_flags`, which drops every "
    "row with a null in any numeric column before it computes anything (F-32). `wide` "
    "has columns with nulls, so the task's skewness came from a subset of the rows "
    "and read 1 or less. The column's own skewness, over all 1 000 values, is 1.16 "
    "for `cont_skew_04` and 1.11 for `cont_skew_19`, above the threshold, so the "
    "check cuts at quantiles."
)
LEFT_SKEW = (
    "The task's skew rule was `skewness > 1`, so a left-skewed column never counted "
    "as skewed: `left_skewed` has skewness -1.89. Equal-width bins crowd the rows "
    "into a few bins for a left tail as much as for a right one, so the check uses "
    "the absolute skewness."
)
BOUNDED_RANGE = (
    "The task chose quantile bins when the range was 3 standard deviations or less. "
    "`subtle_leakage` meets that (range 136.5, standard deviation 46.4, so 2.94) "
    "because its values are bounded and nearly uniform, not because they are crowded. "
    "The middle 98% of them covers 88% of the range and the skewness is 0.83, so "
    "equal-width bins fill evenly and the check suggests them."
)
SIMPLEST_TRANSFORM = (
    "The task recommended the transform with the largest skew reduction, and "
    "Yeo-Johnson, which fits its own power to the column, nearly always wins that. The "
    "check recommends the simplest transform that brings the absolute skewness to 0.5 "
    "or less (log1p, then sqrt, then Box-Cox, then Yeo-Johnson): a plain log or root "
    "is easier to explain, to invert and to keep in a pipeline, and a fitted power "
    "adds a parameter to estimate."
)
NO_EFFECTIVE_TRANSFORM = (
    "`outlier_column` has skewness 5.36 from a few extreme values. No transform "
    "removes a quarter of it: the best, Yeo-Johnson, raises it to 5.82. The task "
    "recommended it anyway, as the best of what it tried even when that made the skew "
    "worse; the check says no transform helps and points to the outliers."
)
UNIMPUTABLE = (
    "The task suggested a fill for every column with nulls and took an unknown type "
    "for continuous. These columns are all null or hold values of mixed types, and "
    "the profile types them `unknown` or `text`; a fill for them would be invented, "
    "so the check gives no advice."
)
MISSING_LEVEL = (
    "At a significant share of nulls the task added an indicator beside the mode. For "
    "a categorical column the check fills with an explicit `missing` level, which is "
    "the indicator: a second column would carry the same information."
)
IRREGULAR_DATES = (
    "`datetime_with_gaps` has no regular frequency (it has gaps), so the previous "
    "value is not a fair fill for a missing date. The task forward-filled it with no "
    "indicator; the check adds an indicator and leaves the gap."
)
MECHANISM = (
    "`missingness_mechanism` finds that whether the value is missing depends on other "
    "columns, which the task never read, so a model of those columns biases the "
    "filled values less than the mean does, and the check adds an indicator for the "
    "pattern."
)


def _each(
    check: str,
    kind: str,
    detail: str,
    reason: str,
    columns: Mapping[str, Sequence[str]],
) -> list[Accepted]:
    """The same difference on the given columns of each dataset."""
    return [
        Accepted(check, dataset, column, kind, detail, reason)
        for dataset, names in columns.items()
        for column in names
    ]


ACCEPTED_ML_PREP: tuple[Accepted, ...] = (
    *_each(
        "imputation_strategy_suggestions",
        "column_read",
        "legacy only",
        UNIMPUTABLE,
        {
            "kitchen5k": [
                "constant_null",
                "empty_vs_nan",
                "mixed_null_types",
                "mixed_type_column",
            ],
        },
    ),
    *_each(
        "imputation_strategy_suggestions",
        "finding",
        "legacy only (info)",
        UNIMPUTABLE,
        {
            "kitchen5k": ["mixed_type_column"],
        },
    ),
    *_each(
        "imputation_strategy_suggestions",
        "finding",
        "legacy only (warn)",
        UNIMPUTABLE,
        {
            "kitchen5k": ["constant_null", "empty_vs_nan", "mixed_null_types"],
        },
    ),
    *_each(
        "imputation_strategy_suggestions",
        "indicator",
        "legacy False, core True",
        IRREGULAR_DATES,
        {
            "kitchen5k": ["datetime_with_gaps"],
        },
    ),
    *_each(
        "imputation_strategy_suggestions",
        "indicator",
        "legacy False, core True",
        MECHANISM,
        {
            "titanic": ["age"],
        },
    ),
    *_each(
        "imputation_strategy_suggestions",
        "indicator",
        "legacy True, core False",
        MISSING_LEVEL,
        {
            "high_missingness": ["device_type"],
            "kitchen5k": ["short_text"],
        },
    ),
    *_each(
        "imputation_strategy_suggestions",
        "strategy",
        "legacy forward_fill, core indicator_only",
        IRREGULAR_DATES,
        {
            "kitchen5k": ["datetime_with_gaps"],
        },
    ),
    *_each(
        "imputation_strategy_suggestions",
        "strategy",
        "legacy mean, core iterative",
        MECHANISM,
        {
            "titanic": ["age"],
        },
    ),
    *_each(
        "imputation_strategy_suggestions",
        "strategy",
        "legacy mean_or_knn_with_indicator, core iterative",
        MECHANISM,
        {
            "high_missingness": ["premium_score"],
        },
    ),
    *_each(
        "imputation_strategy_suggestions",
        "strategy",
        "legacy mode_with_indicator, core constant_missing",
        MISSING_LEVEL,
        {
            "high_missingness": ["device_type"],
            "kitchen5k": ["short_text"],
        },
    ),
    *_each(
        "suggest_categorical_encoding",
        "column_read",
        "core only",
        CODE_COLUMN,
        {
            "kitchen5k": [
                "balanced_target",
                "imbalanced_target",
                "mostly_zeros",
                "multiclass_target",
                "near_zero_variance",
                "perfect_leakage",
                "target_for_leakage",
            ],
            "titanic": ["parch", "pclass", "sibsp", "survived"],
        },
    ),
    *_each(
        "suggest_categorical_encoding",
        "column_read",
        "legacy only",
        NOT_CATEGORICAL,
        {
            "kitchen5k": [
                "base64_encoded",
                "constant_null",
                "constant_string",
                "datetime_with_gaps",
                "email_with_violations",
                "empty_vs_nan",
                "hex_encoded",
                "high_cardinality_uuid",
                "long_text",
                "mixed_null_types",
                "mixed_type_column",
                "normal_date",
                "normal_text",
                "phone_with_violations",
                "transaction_id",
                "user_id",
                "uuid_column",
                "variable_length_text",
            ],
            "sms_spam": ["text"],
            "tiny": ["signup"],
        },
    ),
    *_each(
        "suggest_categorical_encoding",
        "finding",
        "core only (info)",
        CODE_COLUMN,
        {
            "kitchen5k": [
                "balanced_target",
                "imbalanced_target",
                "mostly_zeros",
                "multiclass_target",
                "near_zero_variance",
                "perfect_leakage",
                "target_for_leakage",
            ],
            "titanic": ["parch", "pclass", "sibsp", "survived"],
        },
    ),
    *_each(
        "suggest_categorical_encoding",
        "finding",
        "legacy only (info)",
        NOT_CATEGORICAL,
        {
            "kitchen5k": [
                "constant_null",
                "constant_string",
                "empty_vs_nan",
                "mixed_null_types",
            ],
            "tiny": ["signup"],
        },
    ),
    *_each(
        "suggest_categorical_encoding",
        "finding",
        "legacy only (warn)",
        NOT_CATEGORICAL,
        {
            "kitchen5k": [
                "base64_encoded",
                "datetime_with_gaps",
                "email_with_violations",
                "hex_encoded",
                "high_cardinality_uuid",
                "long_text",
                "mixed_type_column",
                "normal_date",
                "normal_text",
                "phone_with_violations",
                "transaction_id",
                "user_id",
                "uuid_column",
                "variable_length_text",
            ],
            "sms_spam": ["text"],
        },
    ),
    *_each(
        "suggest_categorical_encoding",
        "severity",
        "legacy warn, core info",
        ADVICE_SEVERITY,
        {
            "all_categorical": ["tag"],
            "kitchen5k": [
                "extremely_long_strings",
                "high_cardinality_random",
                "unexpected_object_column",
            ],
            "wide": ["cat_hc"],
        },
    ),
    *_each(
        "suggest_categorical_encoding",
        "strategy",
        "legacy one_hot, core binary",
        BINARY,
        {
            "all_categorical": ["dominant"],
            "kitchen5k": ["control_character_text"],
            "sms_spam": ["label"],
            "titanic": ["alive", "sex"],
        },
    ),
    *_each(
        "suggest_categorical_encoding",
        "strategy",
        "legacy one_hot, core frequency",
        SPARSE,
        {
            "tiny": ["category"],
        },
    ),
    *_each(
        "suggest_numerical_binning",
        "column_read",
        "legacy only",
        CODE_COLUMN_BINNING,
        {
            "kitchen5k": [
                "balanced_target",
                "imbalanced_target",
                "mostly_zeros",
                "multiclass_target",
                "near_zero_variance",
                "perfect_leakage",
                "target_for_leakage",
            ],
            "titanic": ["parch", "pclass", "sibsp", "survived"],
        },
    ),
    *_each(
        "suggest_numerical_binning",
        "column_read",
        "legacy only",
        TOO_FEW_ROWS,
        {
            "tiny": ["x", "y", "z"],
        },
    ),
    *_each(
        "suggest_numerical_binning",
        "finding",
        "legacy only (info)",
        CODE_COLUMN_BINNING,
        {
            "kitchen5k": ["balanced_target", "perfect_leakage", "target_for_leakage"],
            "titanic": ["pclass", "survived"],
        },
    ),
    *_each(
        "suggest_numerical_binning",
        "finding",
        "legacy only (info)",
        TOO_FEW_ROWS,
        {
            "tiny": ["x", "y", "z"],
        },
    ),
    *_each(
        "suggest_numerical_binning",
        "finding",
        "legacy only (warn)",
        CODE_COLUMN_BINNING,
        {
            "kitchen5k": [
                "imbalanced_target",
                "mostly_zeros",
                "multiclass_target",
                "near_zero_variance",
            ],
            "titanic": ["parch", "sibsp"],
        },
    ),
    *_each(
        "suggest_numerical_binning",
        "severity",
        "legacy warn, core info",
        ADVICE_SEVERITY,
        {
            "kitchen5k": [
                "age_with_bounds_issues",
                "extreme_skew",
                "outlier_column",
                "right_skewed",
                "sparse_column",
            ],
            "near_clean": ["purchase_amount"],
            "single_column": ["value"],
            "titanic": ["fare"],
            "wide": [
                "cont_skew_00",
                "cont_skew_01",
                "cont_skew_02",
                "cont_skew_03",
                "cont_skew_05",
                "cont_skew_06",
                "cont_skew_07",
                "cont_skew_08",
                "cont_skew_09",
                "cont_skew_10",
                "cont_skew_11",
                "cont_skew_12",
                "cont_skew_13",
                "cont_skew_14",
                "cont_skew_15",
                "cont_skew_16",
                "cont_skew_17",
                "cont_skew_18",
            ],
        },
    ),
    *_each(
        "suggest_numerical_binning",
        "strategy",
        "legacy equal_width, core quantile",
        LEFT_SKEW,
        {
            "kitchen5k": ["left_skewed"],
        },
    ),
    *_each(
        "suggest_numerical_binning",
        "strategy",
        "legacy equal_width, core quantile",
        LISTWISE_SKEW,
        {
            "wide": ["cont_skew_04", "cont_skew_19"],
        },
    ),
    *_each(
        "suggest_numerical_binning",
        "strategy",
        "legacy quantile, core equal_width",
        BOUNDED_RANGE,
        {
            "kitchen5k": ["subtle_leakage"],
        },
    ),
    *_each(
        "transformation_preview",
        "recommended",
        "legacy yeo_johnson, core None",
        NO_EFFECTIVE_TRANSFORM,
        {
            "kitchen5k": ["outlier_column"],
        },
    ),
    *_each(
        "transformation_preview",
        "recommended",
        "legacy yeo_johnson, core box_cox",
        SIMPLEST_TRANSFORM,
        {
            "kitchen5k": ["extreme_skew"],
            "near_clean": ["purchase_amount"],
            "wide": ["cont_skew_05"],
        },
    ),
    *_each(
        "transformation_preview",
        "recommended",
        "legacy yeo_johnson, core log1p",
        SIMPLEST_TRANSFORM,
        {
            "kitchen5k": ["right_skewed"],
            "single_column": ["value"],
            "titanic": ["fare"],
            "wide": ["cont_skew_04", "cont_skew_08", "cont_skew_15", "cont_skew_17"],
        },
    ),
    *_each(
        "transformation_preview",
        "recommended",
        "legacy yeo_johnson, core sqrt",
        SIMPLEST_TRANSFORM,
        {
            "wide": [
                "cont_skew_00",
                "cont_skew_01",
                "cont_skew_02",
                "cont_skew_03",
                "cont_skew_06",
                "cont_skew_07",
                "cont_skew_09",
                "cont_skew_10",
                "cont_skew_11",
                "cont_skew_12",
                "cont_skew_13",
                "cont_skew_14",
                "cont_skew_16",
                "cont_skew_18",
                "cont_skew_19",
            ],
        },
    ),
)
