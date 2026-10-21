"""The differences between a core check and its legacy task that DSBF accepts.

Each entry names the check, the golden dataset, the column (or none), the kind of
difference and its exact detail, and gives the reason the core check is right. A
difference that is not listed fails ``test_migration_parity``, and so does a listed one
that no longer occurs. Reasons are written by hand.
"""

from __future__ import annotations

from tests.golden.migration.harness import Accepted

CODED_COLUMN = (
    "The profile types an integer column categorical when it has two values, or at "
    "most 20 distinct values under 5% of the rows, so it is a label or a small count "
    "code. `detect_zeros` read every numeric dtype and called its zeros a "
    "concentration; `zeros` reads only numeric columns, where a zero is a "
    "measurement, so a zero in a code is a category and not a finding."
)
CONSTANT_COLUMN = (
    "A column with one value is typed categorical, not numeric, so `zeros` doesn't "
    "read it; constant columns are reported by their own check."
)
FEW_VALUES_COLUMN = (
    "A float column of 11 distinct values under 5% of the rows is typed categorical, "
    "so `zeros` doesn't read it. It has no zeros, so nothing is lost here."
)


def _legacy_only(
    dataset: str, column: str, reason: str, finding: str | None = None
) -> list[Accepted]:
    """The task read ``column`` (and raised a finding about it); the check does not."""
    entries = [
        Accepted("detect_zeros", dataset, column, "column_read", "legacy only", reason)
    ]
    if finding:
        detail = f"legacy only ({finding})"
        entries.append(
            Accepted("detect_zeros", dataset, column, "finding", detail, reason)
        )
    return entries


ALL_NULL_COLUMN = (
    "A column with no values at all tells rows apart no better than a constant one. "
    "`detect_constant_columns` counted distinct non-null values and only called a "
    "column constant when it had exactly one, so it missed a column that has none; "
    "`constant_columns` reports both, and `null_rate` reports the column as missing."
)
SECOND_OF_A_PAIR = (
    "`detect_duplicate_columns` raised its finding on the first column of each pair "
    "only, so the second never had one. `duplicate_columns` raises one finding on the "
    "group, which every column in it shares, and the pairs are the same."
)
TWO_VALUE_STRINGS = (
    "`summarize_boolean_fields` took a two-valued string column for a boolean and "
    "compared it with True and False, so its shares were 0 and it raised no "
    "finding. `dominant_value` reads the most frequent value of any categorical or "
    "boolean column, so it reports the imbalance."
)
CONSTANT_IS_CONSTANT_COLUMNS = (
    "A column of one value is `constant_columns`' to report. `dominant_value` skips "
    "it, where `detect_single_dominant_value` raised a second finding about it."
)
NO_COLUMN_OF_THE_TYPE = (
    "The legacy task read the columns its intent selected, numeric ones included, "
    "and the profile types every column of this dataset numeric, so there is no "
    "categorical column for the check to read and it is not applicable."
)
NO_CATEGORICAL_OR_ID = (
    "`id_columns` also reads numeric columns, for a dense run of whole numbers that "
    "is a row number; `detect_id_columns` read only categorical, text and id columns "
    "and so had nothing to do on a dataset of numeric ones. The check finds no "
    "identifier here."
)
TEXT_TYPED = (
    "The profile types the column text, which `id_columns` reads where the legacy "
    "task, using its own type, did not; nearly all its values differ. The finding is "
    "low confidence for a text column."
)
NUMERIC_ZERO_MASS = (
    "The profile types the column numeric, and a numeric column's mass of one value "
    "is `zeros`' to report. `detect_single_dominant_value` read every column."
)
CODED_FEW_VALUES = (
    "The profile types a column with few distinct values under 5% of the rows (or "
    "one value) categorical, so `near_zero_variance`, which reads numeric columns, "
    "doesn't read it; a column of one value is `constant_columns`', and a few values "
    "are `dominant_value`'s."
)


def _entry(
    task: str, dataset: str, subject: str, kind: str, detail: str, reason: str
) -> Accepted:
    return Accepted(task, dataset, subject, kind, detail, reason)


ACCEPTED: tuple[Accepted, ...] = (
    _entry(
        "summarize_boolean_fields",
        "all_categorical",
        "dominant",
        "finding",
        "core only (warn)",
        TWO_VALUE_STRINGS,
    ),
    _entry(
        "summarize_boolean_fields",
        "all_categorical",
        "is_premium",
        "finding",
        "core only (info)",
        TWO_VALUE_STRINGS,
    ),
    _entry(
        "summarize_boolean_fields",
        "kitchen5k",
        "control_character_text",
        "finding",
        "core only (warn)",
        TWO_VALUE_STRINGS,
    ),
    *(
        _entry("detect_id_columns", dataset, "", "status", status, NO_CATEGORICAL_OR_ID)
        for dataset, status in (
            ("severe_multicollinearity", "legacy not_applicable, core ok"),
            ("single_column", "legacy not_applicable, core ok"),
        )
    ),
    *(
        _entry(
            "detect_single_dominant_value",
            dataset,
            "",
            "status",
            "legacy ok, core not_applicable",
            NO_COLUMN_OF_THE_TYPE,
        )
        for dataset in ("severe_multicollinearity", "single_column")
    ),
    *(
        _entry(
            "detect_constant_columns",
            "kitchen5k",
            column,
            kind,
            detail,
            ALL_NULL_COLUMN,
        )
        for column in ("constant_null", "empty_vs_nan", "mixed_null_types")
        for kind, detail in (
            ("finding", "core only (error)"),
            ("flagged", "core only"),
        )
    ),
    *(
        _entry(
            "detect_duplicate_columns",
            "kitchen5k",
            column,
            "finding",
            "core only (warn)",
            SECOND_OF_A_PAIR,
        )
        for column in ("duplicate_col_3", "empty_vs_nan", "perfect_leakage")
    ),
    _entry(
        "detect_id_columns",
        "kitchen5k",
        "mixed_type_column",
        "finding",
        "core only (warn)",
        TEXT_TYPED,
    ),
    _entry(
        "detect_id_columns",
        "kitchen5k",
        "mixed_type_column",
        "flagged",
        "core only",
        TEXT_TYPED,
    ),
    *(
        _entry(
            "detect_near_zero_variance",
            "kitchen5k",
            column,
            kind,
            detail,
            CODED_FEW_VALUES,
        )
        for column in ("constant_numeric", "near_zero_variance")
        for kind, detail in (
            ("finding", "legacy only (warn)"),
            ("flagged", "legacy only"),
        )
    ),
    *(
        _entry(
            "detect_single_dominant_value",
            "kitchen5k",
            column,
            "finding",
            "legacy only (warn)",
            reason,
        )
        for column, reason in (
            ("constant_numeric", CONSTANT_IS_CONSTANT_COLUMNS),
            ("constant_string", CONSTANT_IS_CONSTANT_COLUMNS),
            ("sparse_column", NUMERIC_ZERO_MASS),
        )
    ),
    *_legacy_only("titanic", "survived", CODED_COLUMN, "warn"),
    *_legacy_only("titanic", "pclass", CODED_COLUMN),
    *_legacy_only("titanic", "sibsp", CODED_COLUMN, "warn"),
    *_legacy_only("titanic", "parch", CODED_COLUMN, "warn"),
    *_legacy_only("kitchen5k", "balanced_target", CODED_COLUMN, "info"),
    *_legacy_only("kitchen5k", "imbalanced_target", CODED_COLUMN, "warn"),
    *_legacy_only("kitchen5k", "mostly_zeros", CODED_COLUMN, "warn"),
    *_legacy_only("kitchen5k", "multiclass_target", CODED_COLUMN, "warn"),
    *_legacy_only("kitchen5k", "perfect_leakage", CODED_COLUMN, "warn"),
    *_legacy_only("kitchen5k", "target_for_leakage", CODED_COLUMN, "warn"),
    *_legacy_only("kitchen5k", "constant_numeric", CONSTANT_COLUMN),
    *_legacy_only("kitchen5k", "near_zero_variance", FEW_VALUES_COLUMN),
)
