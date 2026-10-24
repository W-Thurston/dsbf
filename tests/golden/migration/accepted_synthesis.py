"""What differs between the synthesis checks and the scorers they replace.

See ``accepted.py`` for what an entry is. The comparison is per dimension: its level,
and the columns only one of the two flagged.
A scorer sums up other tasks' results, so most of what differs follows from an
accepted difference of one of them; the rest is the scorers' own design, below.
"""

from __future__ import annotations

from tests.golden.migration.harness import Accepted

DOMINANT_IS_DISTRIBUTION = (
    "A column with one value in 95% of its rows was a usability finding of the legacy"
    " scorer. `dominant_value` declares the distribution dimension (U4-04), so "
    "`quality_dimensions` does not count it; `ml_readiness` lists it under Unusable "
    "features."
)
DUPLICATE_ROWS = (
    "`duplicate_rows` is a validity input of the core scorer, and it counts repeated "
    "rows. The legacy scorer did not read `detect_duplicates`."
)
ERROR_IS_RED = (
    "An error is red. The legacy scorer's docstring says so, but its code gave amber "
    "to every error with under 15% of the columns affected. `null_rate` raises an "
    "error for a column that is mostly missing, and `titanic`'s `deck` (77% missing) "
    "is one."
)
TARGET_LEAKAGE = (
    "kitchen5k plants `perfect_leakage`, `future_leakage` and `subtle_leakage` "
    "against `target_for_leakage`. The core Leakage is `target_leakage`, which flags "
    "the three; the legacy dimension was the feature–feature pairs (the copies and "
    "correlated columns), which are Redundancy."
)
READINESS_ROUTING = (
    "Each legacy readiness dimension carried the blurbs of fixed tasks. The core "
    "routes `near_zero_variance`, `zeros` and `dominant_value` to Unusable features, "
    "where the legacy scorer filed them under Missingness impact, Transformations "
    "needed and Encoding required, and the columns left are those of the component "
    "tasks' accepted differences (`detect_outliers`, `detect_skewness`, "
    "`detect_zeros`, `suggest_categorical_encoding`). `constant_columns`, "
    "`id_columns` and `null_rate` add the all-null and mixed-null columns to Unusable"
    " features."
)
REDUNDANCY_BROADER = (
    "The legacy Redundancy was the VIF above 10, which flagged nothing on this "
    "dataset. The core also counts `duplicate_columns` (the planted copies, and the "
    "all-null columns that equal each other) and the |r| of 0.99 or more pairs of "
    "`associations`, which were the legacy Leakage dimension (F-26)."
)
USABILITY_CORE_ONLY = (
    "`constant_columns` flags the all-null and mixed-null columns and `id_columns` "
    "the mixed-type one, which `detect_constant_columns` and `detect_id_columns` did "
    "not (their accepted differences)."
)
USABILITY_LEGACY_ONLY = (
    "The legacy usability took these through its dominant-value, identifier and "
    "cardinality tasks. The core `dominant_value` declares the distribution "
    "dimension, and `control_character_text` is where `string_anomalies` puts it, in "
    "validity."
)
VALIDITY_BROADER = (
    "The core validity reads `mixed_types`, `format_violations`, `string_anomalies` "
    "and `text_encoding`, which the legacy scorer did not."
)
CONSTANT_IS_USABILITY = (
    "A constant column was a validity finding of the legacy scorer; "
    "`constant_columns` declares usability in the core, where these columns appear. "
    "`imbalanced_target` was flagged by `detect_zeros`' structural-zero rule, which "
    "`zeros` reports as distribution."
)
NO_TARGET = (
    "Only kitchen5k gives the core checks a target. Leakage is a feature that "
    "reproduces the target (F-26), so without one `target_leakage` ends "
    "`not_applicable` and the dimension reads `not assessed`, with no level. The "
    "legacy scorer read the feature–feature pairs of `detect_data_leakage`, which are"
    " redundancy, and showed green when it found none, which said nothing about a "
    "target."
)
LEAKAGE_WAS_REDUNDANCY = (
    "The legacy Leakage dimension was the feature–feature pairs of "
    "`detect_data_leakage` (`base` and `copy_a`, |r| above 0.95). F-26 moved those to"
    " Redundancy, where `associations` reports them as `redundant_columns`, and "
    "Leakage is now target leakage, which this dataset has no target to assess."
)
SAMPLE_SIZE = (
    "`sample_size` reports that the dataset is small for the models it names, and is "
    "a usability input of the core scorer. The legacy scorer did not read "
    "`sample_size_adequacy`."
)
INVISIBLE_AND_NEAR_DUPLICATES = (
    "The core validity reads `string_anomalies` (invisible characters), "
    "`fuzzy_duplicates` (near-duplicate rows) and `duplicate_rows`, which the legacy "
    "scorer did not. `text` is one of the two columns, so a warning on it is above "
    "15% of the columns and the dimension is red."
)
INTEGER_CODES_NOT_FLAGGED = (
    "The legacy readiness dimensions carry the blurbs of `detect_outliers`, "
    "`suggest_numerical_binning` and `detect_zeros` on titanic's integer-coded "
    "columns (`parch`, `pclass`, `sibsp`, `survived`). The core checks raise no "
    "warning for them, which is the accepted difference of those tasks on titanic."
)
CENTRED_VIF = (
    "The legacy Redundancy was the task's VIF above 10, computed on uncentred "
    "columns, which inflates the VIF of a column with a large mean, so clean columns "
    "were flagged. The core `collinearity` centres them, and its accepted differences"
    " from `detect_collinear_features` are these columns."
)

ACCEPTED_SYNTHESIS: tuple[Accepted, ...] = (
    Accepted(
        "data_quality_scorer",
        "all_categorical",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "all_categorical",
        "usability",
        "legacy_only",
        "1 columns: dominant",
        DOMINANT_IS_DISTRIBUTION,
    ),
    Accepted(
        "data_quality_scorer",
        "all_categorical",
        "usability",
        "level",
        "legacy red, core amber",
        DOMINANT_IS_DISTRIBUTION,
    ),
    Accepted(
        "ml_readiness_scorer",
        "all_categorical",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "clean",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "clean",
        "redundancy",
        "legacy_only",
        "4 columns: age, income, ...",
        CENTRED_VIF,
    ),
    Accepted(
        "data_quality_scorer",
        "clean",
        "redundancy",
        "level",
        "legacy red, core green",
        CENTRED_VIF,
    ),
    Accepted(
        "ml_readiness_scorer",
        "clean",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "high_missingness",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "high_missingness",
        "redundancy",
        "legacy_only",
        "1 columns: age",
        CENTRED_VIF,
    ),
    Accepted(
        "data_quality_scorer",
        "high_missingness",
        "redundancy",
        "level",
        "legacy amber, core green",
        CENTRED_VIF,
    ),
    Accepted(
        "data_quality_scorer",
        "high_missingness",
        "validity",
        "level",
        "legacy green, core blue",
        DUPLICATE_ROWS,
    ),
    Accepted(
        "ml_readiness_scorer",
        "high_missingness",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "kitchen5k",
        "completeness",
        "level",
        "legacy amber, core red",
        ERROR_IS_RED,
    ),
    Accepted(
        "data_quality_scorer",
        "kitchen5k",
        "leakage",
        "core_only",
        "2 columns: datetime_with_gaps, future_leakage",
        TARGET_LEAKAGE,
    ),
    Accepted(
        "data_quality_scorer",
        "kitchen5k",
        "leakage",
        "legacy_only",
        "8 columns: corr_base, duplicate_col_1, ...",
        TARGET_LEAKAGE,
    ),
    Accepted(
        "data_quality_scorer",
        "kitchen5k",
        "redundancy",
        "core_only",
        "13 columns: constant_null, corr_base, ...",
        REDUNDANCY_BROADER,
    ),
    Accepted(
        "data_quality_scorer",
        "kitchen5k",
        "redundancy",
        "level",
        "legacy green, core red",
        REDUNDANCY_BROADER,
    ),
    Accepted(
        "data_quality_scorer",
        "kitchen5k",
        "usability",
        "core_only",
        "4 columns: constant_null, empty_vs_nan, ...",
        USABILITY_CORE_ONLY,
    ),
    Accepted(
        "data_quality_scorer",
        "kitchen5k",
        "usability",
        "legacy_only",
        "4 columns: control_character_text, imbalanced_boolean, ...",
        USABILITY_LEGACY_ONLY,
    ),
    Accepted(
        "data_quality_scorer",
        "kitchen5k",
        "validity",
        "core_only",
        "6 columns: ambiguous_booleans, control_character_text, ...",
        VALIDITY_BROADER,
    ),
    Accepted(
        "data_quality_scorer",
        "kitchen5k",
        "validity",
        "legacy_only",
        "3 columns: constant_numeric, constant_string, ...",
        CONSTANT_IS_USABILITY,
    ),
    Accepted(
        "ml_readiness_scorer",
        "kitchen5k",
        "encoding",
        "legacy_only",
        "4 columns: imbalanced_boolean, imbalanced_target, ...",
        READINESS_ROUTING,
    ),
    Accepted(
        "ml_readiness_scorer",
        "kitchen5k",
        "leakage",
        "core_only",
        "2 columns: datetime_with_gaps, future_leakage",
        TARGET_LEAKAGE,
    ),
    Accepted(
        "ml_readiness_scorer",
        "kitchen5k",
        "leakage",
        "legacy_only",
        "8 columns: corr_base, duplicate_col_1, ...",
        TARGET_LEAKAGE,
    ),
    Accepted(
        "ml_readiness_scorer",
        "kitchen5k",
        "missingness",
        "legacy_only",
        "2 columns: constant_numeric, near_zero_variance",
        READINESS_ROUTING,
    ),
    Accepted(
        "ml_readiness_scorer",
        "kitchen5k",
        "transformations",
        "legacy_only",
        "7 columns: balanced_target, imbalanced_target, ...",
        READINESS_ROUTING,
    ),
    Accepted(
        "ml_readiness_scorer",
        "kitchen5k",
        "unusable",
        "core_only",
        "4 columns: constant_null, empty_vs_nan, ...",
        READINESS_ROUTING,
    ),
    Accepted(
        "ml_readiness_scorer",
        "kitchen5k",
        "unusable",
        "legacy_only",
        "3 columns: multiclass_target, perfect_leakage, ...",
        READINESS_ROUTING,
    ),
    Accepted(
        "data_quality_scorer",
        "near_clean",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "near_clean",
        "redundancy",
        "legacy_only",
        "3 columns: age, income, ...",
        CENTRED_VIF,
    ),
    Accepted(
        "data_quality_scorer",
        "near_clean",
        "redundancy",
        "level",
        "legacy red, core green",
        CENTRED_VIF,
    ),
    Accepted(
        "ml_readiness_scorer",
        "near_clean",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "severe_multicollinearity",
        "completeness",
        "level",
        "legacy amber, core red",
        ERROR_IS_RED,
    ),
    Accepted(
        "data_quality_scorer",
        "severe_multicollinearity",
        "leakage",
        "legacy_only",
        "2 columns: base, copy_a",
        LEAKAGE_WAS_REDUNDANCY,
    ),
    Accepted(
        "data_quality_scorer",
        "severe_multicollinearity",
        "leakage",
        "level",
        "legacy red, core not assessed",
        LEAKAGE_WAS_REDUNDANCY,
    ),
    Accepted(
        "data_quality_scorer",
        "severe_multicollinearity",
        "redundancy",
        "legacy_only",
        "2 columns: signal, sparse_c",
        CENTRED_VIF,
    ),
    Accepted(
        "ml_readiness_scorer",
        "severe_multicollinearity",
        "leakage",
        "legacy_only",
        "2 columns: base, copy_a",
        LEAKAGE_WAS_REDUNDANCY,
    ),
    Accepted(
        "ml_readiness_scorer",
        "severe_multicollinearity",
        "leakage",
        "level",
        "legacy red, core not assessed",
        LEAKAGE_WAS_REDUNDANCY,
    ),
    Accepted(
        "data_quality_scorer",
        "single_column",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "single_column",
        "usability",
        "level",
        "legacy green, core blue",
        SAMPLE_SIZE,
    ),
    Accepted(
        "data_quality_scorer",
        "single_column",
        "validity",
        "level",
        "legacy green, core amber",
        DUPLICATE_ROWS,
    ),
    Accepted(
        "ml_readiness_scorer",
        "single_column",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "sms_spam",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "sms_spam",
        "validity",
        "core_only",
        "1 columns: text",
        INVISIBLE_AND_NEAR_DUPLICATES,
    ),
    Accepted(
        "data_quality_scorer",
        "sms_spam",
        "validity",
        "level",
        "legacy green, core red",
        INVISIBLE_AND_NEAR_DUPLICATES,
    ),
    Accepted(
        "ml_readiness_scorer",
        "sms_spam",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "tiny",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "tiny",
        "redundancy",
        "legacy_only",
        "3 columns: x, y, ...",
        CENTRED_VIF,
    ),
    Accepted(
        "data_quality_scorer",
        "tiny",
        "redundancy",
        "level",
        "legacy red, core green",
        CENTRED_VIF,
    ),
    Accepted(
        "data_quality_scorer",
        "tiny",
        "usability",
        "level",
        "legacy green, core amber",
        SAMPLE_SIZE,
    ),
    Accepted(
        "ml_readiness_scorer",
        "tiny",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "titanic",
        "completeness",
        "level",
        "legacy amber, core red",
        ERROR_IS_RED,
    ),
    Accepted(
        "data_quality_scorer",
        "titanic",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "titanic",
        "usability",
        "level",
        "legacy green, core blue",
        SAMPLE_SIZE,
    ),
    Accepted(
        "data_quality_scorer",
        "titanic",
        "validity",
        "level",
        "legacy green, core amber",
        DUPLICATE_ROWS,
    ),
    Accepted(
        "ml_readiness_scorer",
        "titanic",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "ml_readiness_scorer",
        "titanic",
        "transformations",
        "legacy_only",
        "4 columns: parch, pclass, ...",
        INTEGER_CODES_NOT_FLAGGED,
    ),
    Accepted(
        "ml_readiness_scorer",
        "titanic",
        "unusable",
        "legacy_only",
        "3 columns: parch, sibsp, ...",
        INTEGER_CODES_NOT_FLAGGED,
    ),
    Accepted(
        "ml_readiness_scorer",
        "titanic",
        "unusable",
        "level",
        "legacy amber, core green",
        INTEGER_CODES_NOT_FLAGGED,
    ),
    Accepted(
        "data_quality_scorer",
        "wide",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
    Accepted(
        "data_quality_scorer",
        "wide",
        "redundancy",
        "legacy_only",
        "30 columns: cont_clean_01, cont_clean_04, ...",
        CENTRED_VIF,
    ),
    Accepted(
        "data_quality_scorer",
        "wide",
        "redundancy",
        "level",
        "legacy red, core green",
        CENTRED_VIF,
    ),
    Accepted(
        "ml_readiness_scorer",
        "wide",
        "leakage",
        "level",
        "legacy green, core not assessed",
        NO_TARGET,
    ),
)
