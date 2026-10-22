"""What differs between the missingness and string structure checks and their tasks.

See ``accepted.py`` for what an entry is. The rows are what ``test_migration_parity``
printed for the golden datasets, grouped by the reason each is right. The null counts
of every column both analysed agree on all eleven datasets, and so do the counts of
whitespace-padded and invisible-character values, which is why none is listed.
"""

from __future__ import annotations

from collections.abc import Iterable

from tests.golden.migration.harness import Accepted

NO_MISSING = (
    "No column has enough nulls to analyse: at least 1% of the rows and 8 nulls "
    "(`min_null_share`, `min_group_n`). The task returned an empty successful "
    "result; the check ends not applicable and names the thresholds."
)
LITTLE = (
    "The task's \"Little's MCAR test\" compared the means of each missing-data "
    "pattern with the overall means and summed the squared differences over each "
    "pattern's variance, counting the observed columns of every pattern as degrees "
    "of freedom. That is not Little's statistic (which weights by the covariance "
    "estimated by expectation-maximisation) and the sum does not follow a chi- "
    "squared with those degrees of freedom, so its p-value says nothing reliable "
    "about MCAR. The check has no dataset-level test: it tests each column's nulls "
    "against every other column and adjusts all the p-values together."
)
UNADJUSTED = (
    "The task ran 72 correlations and 67 Mann-Whitney tests per column, none "
    "corrected for the number run, and called a column consistent with MAR when one "
    "was significant at 0.05: two to six were, in every column, which is what "
    "chance gives (3.4 expected). The generator makes the nulls of `wide`'s "
    "`null_00` to `null_07` completely at random. The check runs 784 tests, adjusts "
    "them together (Benjamini-Hochberg) and requires an effect of 0.1; the smallest "
    "adjusted p is 0.65. In kitchen5k the task counted `duplicate_col_1` to "
    "`duplicate_col_3`, identical columns, as three confirmations of effects of "
    "0.055 and under."
)
CATEGORICAL = (
    "The task compared the null indicator with numeric columns only (point-biserial "
    "correlation and Mann-Whitney); it never read a categorical one. The check also "
    "tests levels by chi-squared with bias-corrected Cramer's V, and the nulls "
    "follow a categorical column: `channel` for `device_type` (V = 0.89) and "
    "`plan_type` for `premium_score` (V = 0.85) in `high_missingness`; "
    "`control_character_text` for `mixed_type_column` (V = 0.43) and "
    "`datetime_with_gaps` (V = 1.0, whose 104 nulls are the 104 rows of the level "
    "`text withcontrols`) in kitchen5k."
)
SEVERITY_WARN = (
    "A column that is over 20% null and depends strongly (an effect of 0.3 or more) "
    "on another is a warning, since the rows with a null are not a random subset "
    "and dropping them biases the rest. The task reported `info` when it found no "
    "numeric evidence of a mechanism."
)
SEVERITY_INFO = (
    "The task warned whenever it found no evidence of a mechanism "
    '("Indeterminate"). For these two columns the check finds a dependence, on a '
    "column the task never read, and reports it at `info` because under 20% of the "
    "values are null."
)
ALL_NULL = (
    "Every value of the column is null (an empty CSV field is null), so there is "
    "nothing to compare; the check lists it in `skipped_columns` with the reason. "
    "The task analysed it and warned that the mechanism was indeterminate."
)
NO_STRINGS = (
    "The data has no string column with at least 10 values (the check's "
    "`min_values`). The task returned an empty successful result; the check ends "
    "not applicable."
)
CONTENT = (
    "The task counted the Python types of the values, and a CSV holds only strings, "
    "so in the engine's run it finds one type and reports nothing. (In the legacy "
    "snapshots it flagged `deck` on titanic and `device_type` and `notes` on "
    "`high_missingness`: strings with a float `NaN` for each null, which is a "
    "second type.) The check classifies what each string reads as: "
    "`mixed_type_column` is 86% numbers and 14% `STRING_n`; `mixed_boolean_types` "
    "and `ambiguous_booleans` write booleans as `True`/`False`/`yes`/`no` beside "
    "`1` and `0`."
)
PLAIN_WORDS = (
    "The task called a column base64 when its first 50 values were made of letters, "
    "digits, `+`, `/` and `=`, which every plain word is, so it reported `sex`, "
    "`embarked`, `class` and `who` on titanic and each categorical column of the "
    "others. The check needs a length of at least 12 that is a multiple of 4, text "
    "that decodes to printable ASCII or both cases with a digit or a padding "
    "character, so no word matches."
)
IDENTIFIERS = (
    "The task read categorical and text columns, and the profile types these three "
    "identifiers (over 90% distinct), so it never looked at them. The check reads "
    "identifiers too: `high_cardinality_uuid` and `uuid_column` are UUIDs, and "
    "`base64_encoded` holds base64 of `data_n`."
)
REGEX_STATUS = (
    "The task needs a column the legacy inference types text, and a configured "
    "pattern (there is none by default), so it was not applicable. The check also "
    "reads every string column for dates written in several formats, so it ends ok "
    "with nothing to report."
)
MIXED_DATES = (
    "The check reports a column that writes dates in several formats. "
    "`inconsistent_datetime` holds four, in equal shares (`2023-01-15`, "
    "`15/01/2023`, `Jan 15 2023`, `20230115`); the task had no such test."
)
RULES = (
    "The golden run gives `format_violations` two rules for this dataset: a label "
    "is `ham` or `spam`, and a message names no link. The task had no pattern "
    "configured. 106 messages hold a link."
)
PROSE = (
    "The `text` column is prose (the profile types it `text`; the task called it an "
    "id and still read it as strings). Capitalisation and length vary by nature in "
    "free text, so the check applies neither the case nor the length test to it; "
    "the task reported one set of messages that differ only by case and 19 messages "
    "beyond its length fence of 380 characters."
)


def _group(
    check: str, dataset: str, reason: str, rows: Iterable[tuple[str, str, str]]
) -> list[Accepted]:
    return [Accepted(check, dataset, s, k, d, reason) for s, k, d in rows]


ACCEPTED_MISSINGNESS_STRINGS: tuple[Accepted, ...] = (
    *_group(
        "detect_encoded_columns",
        "all_categorical",
        PLAIN_WORDS,
        [
            ("category", "column_read", "legacy only"),
            ("category", "finding", "legacy only (info)"),
            ("color", "column_read", "legacy only"),
            ("color", "finding", "legacy only (info)"),
            ("dominant", "column_read", "legacy only"),
            ("dominant", "finding", "legacy only (info)"),
            ("material", "column_read", "legacy only"),
            ("material", "finding", "legacy only (info)"),
            ("region", "column_read", "legacy only"),
            ("region", "finding", "legacy only (info)"),
            ("size", "column_read", "legacy only"),
            ("size", "finding", "legacy only (info)"),
        ],
    ),
    *_group(
        "detect_encoded_columns",
        "clean",
        PLAIN_WORDS,
        [
            ("department", "column_read", "legacy only"),
            ("department", "finding", "legacy only (info)"),
            ("plan_type", "column_read", "legacy only"),
            ("plan_type", "finding", "legacy only (info)"),
            ("region", "column_read", "legacy only"),
            ("region", "finding", "legacy only (info)"),
        ],
    ),
    *_group(
        "detect_encoded_columns",
        "high_missingness",
        PLAIN_WORDS,
        [
            ("channel", "column_read", "legacy only"),
            ("channel", "finding", "legacy only (info)"),
            ("device_type", "column_read", "legacy only"),
            ("device_type", "finding", "legacy only (info)"),
            ("plan_type", "column_read", "legacy only"),
            ("plan_type", "finding", "legacy only (info)"),
            ("region", "column_read", "legacy only"),
            ("region", "finding", "legacy only (info)"),
        ],
    ),
    *_group(
        "detect_encoded_columns",
        "kitchen5k",
        IDENTIFIERS,
        [
            ("base64_encoded", "column_read", "core only"),
            ("base64_encoded", "finding", "core only (info)"),
            ("high_cardinality_uuid", "column_read", "core only"),
            ("high_cardinality_uuid", "finding", "core only (info)"),
            ("uuid_column", "column_read", "core only"),
            ("uuid_column", "finding", "core only (info)"),
        ],
    ),
    *_group(
        "detect_encoded_columns",
        "kitchen5k",
        PLAIN_WORDS,
        [
            ("ambiguous_booleans", "column_read", "legacy only"),
            ("ambiguous_booleans", "finding", "legacy only (info)"),
            ("mixed_boolean_types", "column_read", "legacy only"),
            ("mixed_boolean_types", "finding", "legacy only (info)"),
            ("normal_categorical", "column_read", "legacy only"),
            ("normal_categorical", "finding", "legacy only (info)"),
            ("short_text", "column_read", "legacy only"),
            ("short_text", "finding", "legacy only (info)"),
        ],
    ),
    *_group(
        "detect_encoded_columns",
        "near_clean",
        PLAIN_WORDS,
        [
            ("plan_type", "column_read", "legacy only"),
            ("plan_type", "finding", "legacy only (info)"),
            ("region", "column_read", "legacy only"),
            ("region", "finding", "legacy only (info)"),
            ("segment", "column_read", "legacy only"),
            ("segment", "finding", "legacy only (info)"),
        ],
    ),
    *_group(
        "detect_encoded_columns",
        "sms_spam",
        PLAIN_WORDS,
        [
            ("label", "column_read", "legacy only"),
            ("label", "finding", "legacy only (info)"),
        ],
    ),
    *_group(
        "detect_encoded_columns",
        "tiny",
        PLAIN_WORDS,
        [
            ("category", "column_read", "legacy only"),
            ("category", "finding", "legacy only (info)"),
        ],
    ),
    *_group(
        "detect_encoded_columns",
        "titanic",
        PLAIN_WORDS,
        [
            ("alive", "column_read", "legacy only"),
            ("alive", "finding", "legacy only (info)"),
            ("class", "column_read", "legacy only"),
            ("class", "finding", "legacy only (info)"),
            ("deck", "column_read", "legacy only"),
            ("deck", "finding", "legacy only (info)"),
            ("embark_town", "column_read", "legacy only"),
            ("embark_town", "finding", "legacy only (info)"),
            ("embarked", "column_read", "legacy only"),
            ("embarked", "finding", "legacy only (info)"),
            ("sex", "column_read", "legacy only"),
            ("sex", "finding", "legacy only (info)"),
            ("who", "column_read", "legacy only"),
            ("who", "finding", "legacy only (info)"),
        ],
    ),
    *_group(
        "detect_encoded_columns",
        "wide",
        PLAIN_WORDS,
        [
            ("cat_bal_00", "column_read", "legacy only"),
            ("cat_bal_00", "finding", "legacy only (info)"),
            ("cat_bal_01", "column_read", "legacy only"),
            ("cat_bal_01", "finding", "legacy only (info)"),
            ("cat_bal_02", "column_read", "legacy only"),
            ("cat_bal_02", "finding", "legacy only (info)"),
            ("cat_bal_03", "column_read", "legacy only"),
            ("cat_bal_03", "finding", "legacy only (info)"),
            ("cat_bal_04", "column_read", "legacy only"),
            ("cat_bal_04", "finding", "legacy only (info)"),
            ("cat_bal_05", "column_read", "legacy only"),
            ("cat_bal_05", "finding", "legacy only (info)"),
            ("cat_bal_06", "column_read", "legacy only"),
            ("cat_bal_06", "finding", "legacy only (info)"),
            ("cat_bal_07", "column_read", "legacy only"),
            ("cat_bal_07", "finding", "legacy only (info)"),
            ("cat_bal_08", "column_read", "legacy only"),
            ("cat_bal_08", "finding", "legacy only (info)"),
            ("cat_bal_09", "column_read", "legacy only"),
            ("cat_bal_09", "finding", "legacy only (info)"),
            ("cat_bal_10", "column_read", "legacy only"),
            ("cat_bal_10", "finding", "legacy only (info)"),
            ("cat_bal_11", "column_read", "legacy only"),
            ("cat_bal_11", "finding", "legacy only (info)"),
            ("cat_bal_12", "column_read", "legacy only"),
            ("cat_bal_12", "finding", "legacy only (info)"),
            ("cat_bal_13", "column_read", "legacy only"),
            ("cat_bal_13", "finding", "legacy only (info)"),
            ("cat_bal_14", "column_read", "legacy only"),
            ("cat_bal_14", "finding", "legacy only (info)"),
            ("cat_bal_15", "column_read", "legacy only"),
            ("cat_bal_15", "finding", "legacy only (info)"),
            ("cat_bal_16", "column_read", "legacy only"),
            ("cat_bal_16", "finding", "legacy only (info)"),
            ("cat_bal_17", "column_read", "legacy only"),
            ("cat_bal_17", "finding", "legacy only (info)"),
            ("cat_bal_18", "column_read", "legacy only"),
            ("cat_bal_18", "finding", "legacy only (info)"),
            ("cat_bal_19", "column_read", "legacy only"),
            ("cat_bal_19", "finding", "legacy only (info)"),
            ("cat_bal_20", "column_read", "legacy only"),
            ("cat_bal_20", "finding", "legacy only (info)"),
            ("cat_bal_21", "column_read", "legacy only"),
            ("cat_bal_21", "finding", "legacy only (info)"),
            ("cat_bal_22", "column_read", "legacy only"),
            ("cat_bal_22", "finding", "legacy only (info)"),
            ("cat_bal_23", "column_read", "legacy only"),
            ("cat_bal_23", "finding", "legacy only (info)"),
            ("cat_bal_24", "column_read", "legacy only"),
            ("cat_bal_24", "finding", "legacy only (info)"),
            ("cat_bal_25", "column_read", "legacy only"),
            ("cat_bal_25", "finding", "legacy only (info)"),
        ],
    ),
    *_group(
        "detect_mixed_type_columns",
        "kitchen5k",
        CONTENT,
        [
            ("ambiguous_booleans", "column_read", "core only"),
            ("ambiguous_booleans", "finding", "core only (warn)"),
            ("mixed_boolean_types", "column_read", "core only"),
            ("mixed_boolean_types", "finding", "core only (warn)"),
            ("mixed_type_column", "column_read", "core only"),
            ("mixed_type_column", "finding", "core only (warn)"),
        ],
    ),
    *_group(
        "detect_mixed_type_columns",
        "severe_multicollinearity",
        NO_STRINGS,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "detect_mixed_type_columns",
        "single_column",
        NO_STRINGS,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "detect_regex_format_violations",
        "all_categorical",
        REGEX_STATUS,
        [
            ("", "status", "legacy not_applicable, core ok"),
        ],
    ),
    *_group(
        "detect_regex_format_violations",
        "clean",
        REGEX_STATUS,
        [
            ("", "status", "legacy not_applicable, core ok"),
        ],
    ),
    *_group(
        "detect_regex_format_violations",
        "high_missingness",
        REGEX_STATUS,
        [
            ("", "status", "legacy not_applicable, core ok"),
        ],
    ),
    *_group(
        "detect_regex_format_violations",
        "kitchen5k",
        MIXED_DATES,
        [
            ("inconsistent_datetime", "finding", "core only (warn)"),
        ],
    ),
    *_group(
        "detect_regex_format_violations",
        "near_clean",
        REGEX_STATUS,
        [
            ("", "status", "legacy not_applicable, core ok"),
        ],
    ),
    *_group(
        "detect_regex_format_violations",
        "sms_spam",
        RULES,
        [
            ("text", "column_read", "core only"),
            ("text", "finding", "core only (warn)"),
        ],
    ),
    *_group(
        "detect_regex_format_violations",
        "tiny",
        REGEX_STATUS,
        [
            ("", "status", "legacy not_applicable, core ok"),
        ],
    ),
    *_group(
        "detect_regex_format_violations",
        "titanic",
        REGEX_STATUS,
        [
            ("", "status", "legacy not_applicable, core ok"),
        ],
    ),
    *_group(
        "detect_regex_format_violations",
        "wide",
        REGEX_STATUS,
        [
            ("", "status", "legacy not_applicable, core ok"),
        ],
    ),
    *_group(
        "detect_string_anomalies",
        "sms_spam",
        PROSE,
        [
            ("text", "length_outliers", "legacy only"),
            ("text", "mixed_case", "legacy only"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "all_categorical",
        NO_MISSING,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "clean",
        NO_MISSING,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "high_missingness",
        CATEGORICAL,
        [
            (
                "device_type",
                "dependence",
                "legacy no MAR evidence, core depends_on_observed",
            ),
            (
                "premium_score",
                "dependence",
                "legacy no MAR evidence, core depends_on_observed",
            ),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "high_missingness",
        LITTLE,
        [
            ("", "little_mcar_test", "legacy ran it, core has no such test"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "high_missingness",
        SEVERITY_WARN,
        [
            ("device_type", "severity", "legacy info, core warn"),
            ("premium_score", "severity", "legacy info, core warn"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "kitchen5k",
        ALL_NULL,
        [
            ("constant_null", "column_read", "legacy only"),
            ("constant_null", "finding", "legacy only (warn)"),
            ("empty_vs_nan", "column_read", "legacy only"),
            ("empty_vs_nan", "finding", "legacy only (warn)"),
            ("mixed_null_types", "column_read", "legacy only"),
            ("mixed_null_types", "finding", "legacy only (warn)"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "kitchen5k",
        CATEGORICAL,
        [
            (
                "datetime_with_gaps",
                "dependence",
                "legacy no MAR evidence, core depends_on_observed",
            ),
            (
                "mixed_type_column",
                "dependence",
                "legacy no MAR evidence, core depends_on_observed",
            ),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "kitchen5k",
        SEVERITY_INFO,
        [
            ("datetime_with_gaps", "severity", "legacy warn, core info"),
            ("mixed_type_column", "severity", "legacy warn, core info"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "kitchen5k",
        UNADJUSTED,
        [
            (
                "inconsistent_datetime",
                "dependence",
                "legacy MAR, core no_dependence_found",
            ),
            ("short_text", "dependence", "legacy MAR, core no_dependence_found"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "near_clean",
        LITTLE,
        [
            ("", "little_mcar_test", "legacy ran it, core has no such test"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "severe_multicollinearity",
        LITTLE,
        [
            ("", "little_mcar_test", "legacy ran it, core has no such test"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "single_column",
        NO_MISSING,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "sms_spam",
        NO_MISSING,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "tiny",
        NO_MISSING,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "titanic",
        LITTLE,
        [
            ("", "little_mcar_test", "legacy ran it, core has no such test"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "titanic",
        SEVERITY_WARN,
        [
            ("deck", "severity", "legacy info, core warn"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "wide",
        LITTLE,
        [
            ("", "little_mcar_test", "legacy ran it, core has no such test"),
        ],
    ),
    *_group(
        "missingness_mechanism_analysis",
        "wide",
        UNADJUSTED,
        [
            ("null_00", "dependence", "legacy MAR, core no_dependence_found"),
            ("null_01", "dependence", "legacy MAR, core no_dependence_found"),
            ("null_02", "dependence", "legacy MAR, core no_dependence_found"),
            ("null_03", "dependence", "legacy MAR, core no_dependence_found"),
            ("null_04", "dependence", "legacy MAR, core no_dependence_found"),
            ("null_05", "dependence", "legacy MAR, core no_dependence_found"),
            ("null_06", "dependence", "legacy MAR, core no_dependence_found"),
            ("null_07", "dependence", "legacy MAR, core no_dependence_found"),
        ],
    ),
)
