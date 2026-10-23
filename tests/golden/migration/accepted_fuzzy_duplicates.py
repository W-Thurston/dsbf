"""What differs between the fuzzy duplicate check and its task.

See ``accepted.py`` for what an entry is. The rows are what ``test_migration_parity``
printed for the golden datasets, grouped by the reason each is right. The task scored
the concatenated row string with `difflib` and kept pairs from 0.85 that shared any
token; the check scores each column and weights the columns by their entropy.
"""

from __future__ import annotations

from collections.abc import Iterable

from tests.golden.migration.harness import Accepted

NO_TEXT = (
    "No column has more than 20 distinct string values, so the check has no text to "
    "compare rows by and ends not applicable. The task compared the concatenated "
    "values of the label columns (sex, class, embarked and the like); two rows "
    "agreed on them whenever two people shared a few labels, so what it flagged was "
    "the labels, not a repeated record (F-35)."
)
LABELS = (
    "The task put every string-typed column in the row string, labels included: "
    "columns of 20 or fewer distinct values, codes the profile types categorical, "
    "booleans and dates written as text. They say what kind of row it is, not which "
    "one, so the check leaves them out and lists them in `skipped_columns`."
)
LABEL_PAIRS = (
    "Eight of the nine columns hold two to six values (labels, booleans and codes), "
    "so the task's row string was mostly labels: at 0.85 any two rows that share a "
    "few of them look alike, and it flagged 1 651 pairs. That is the finding F-35 "
    "describes. The check compares the one column with names in it, `tag` (581 "
    "distinct values, every pair scored), and no two differ by 8% or less."
)
EXACT_PAIRS = (
    "The five pairs the task flagged are rows repeated exactly in the string "
    "columns (similarity 1.0, no column differs; the generator appends 5% exact "
    "duplicate rows) whose copies both fell in the task's sample of 5 000 rows. The "
    "check compares distinct records, so a row repeated in every column it reads is "
    "an exact repeat, which it counts and `duplicate_rows` reports, and not a pair."
)
SMS_PAIRS = (
    "The task compared a random sample of 5 000 of the 5 574 messages and its "
    "blocking stopped at 50 000 candidate pairs, so it found five of the pairs. The "
    "check compares all 5 171 distinct messages, scoring the 178 744 pairs that "
    "share a rare token, and finds 217 pairs at 0.92 or more, the closest differing "
    'only by case or spacing ("Sorry, I\'ll call later in meeting." and "... In '
    'meeting.") or by a changed word or number in a long spam message. It is a '
    "warning because 5.7% of the rows are in a pair."
)


def _group(
    dataset: str, reason: str, rows: Iterable[tuple[str, str, str]]
) -> list[Accepted]:
    return [
        Accepted("fuzzy_duplicate_detection", dataset, s, k, d, reason)
        for s, k, d in rows
    ]


NO_TEXT_STATUS = [("", "status", "legacy ok, core not_applicable")]

ACCEPTED_FUZZY_DUPLICATES: tuple[Accepted, ...] = (
    *(
        accepted
        for dataset in ("clean", "tiny", "near_clean", "high_missingness", "titanic")
        for accepted in _group(dataset, NO_TEXT, NO_TEXT_STATUS)
    ),
    *_group(
        "all_categorical",
        LABEL_PAIRS,
        [
            ("", "finding", "legacy warn, core None"),
            ("", "pairs_flagged", "legacy 1651, core 0"),
        ],
    ),
    *_group(
        "all_categorical",
        LABELS,
        [
            ("category", "column_read", "legacy only"),
            ("color", "column_read", "legacy only"),
            ("dominant", "column_read", "legacy only"),
            ("is_active", "column_read", "legacy only"),
            ("is_premium", "column_read", "legacy only"),
            ("material", "column_read", "legacy only"),
            ("region", "column_read", "legacy only"),
            ("size", "column_read", "legacy only"),
        ],
    ),
    *_group(
        "wide",
        LABELS,
        [
            ("bool_00", "column_read", "legacy only"),
            ("bool_01", "column_read", "legacy only"),
            ("bool_02", "column_read", "legacy only"),
            ("bool_03", "column_read", "legacy only"),
            ("bool_04", "column_read", "legacy only"),
            ("cat_bal_00", "column_read", "legacy only"),
            ("cat_bal_01", "column_read", "legacy only"),
            ("cat_bal_02", "column_read", "legacy only"),
            ("cat_bal_03", "column_read", "legacy only"),
            ("cat_bal_04", "column_read", "legacy only"),
            ("cat_bal_05", "column_read", "legacy only"),
            ("cat_bal_06", "column_read", "legacy only"),
            ("cat_bal_07", "column_read", "legacy only"),
            ("cat_bal_08", "column_read", "legacy only"),
            ("cat_bal_09", "column_read", "legacy only"),
            ("cat_bal_10", "column_read", "legacy only"),
            ("cat_bal_11", "column_read", "legacy only"),
            ("cat_bal_12", "column_read", "legacy only"),
            ("cat_bal_13", "column_read", "legacy only"),
            ("cat_bal_14", "column_read", "legacy only"),
            ("cat_bal_15", "column_read", "legacy only"),
            ("cat_bal_16", "column_read", "legacy only"),
            ("cat_bal_17", "column_read", "legacy only"),
            ("cat_bal_18", "column_read", "legacy only"),
            ("cat_bal_19", "column_read", "legacy only"),
            ("cat_bal_20", "column_read", "legacy only"),
            ("cat_bal_21", "column_read", "legacy only"),
            ("cat_bal_22", "column_read", "legacy only"),
            ("cat_bal_23", "column_read", "legacy only"),
            ("cat_bal_24", "column_read", "legacy only"),
            ("cat_bal_25", "column_read", "legacy only"),
        ],
    ),
    *_group(
        "kitchen5k",
        EXACT_PAIRS,
        [
            ("", "finding", "legacy info, core None"),
            ("", "pairs_flagged", "legacy 5, core 0"),
        ],
    ),
    *_group(
        "kitchen5k",
        LABELS,
        [
            ("ambiguous_booleans", "column_read", "legacy only"),
            ("balanced_target", "column_read", "legacy only"),
            ("constant_numeric", "column_read", "legacy only"),
            ("constant_string", "column_read", "legacy only"),
            ("control_character_text", "column_read", "legacy only"),
            ("datetime_with_gaps", "column_read", "legacy only"),
            ("future_leakage", "column_read", "legacy only"),
            ("imbalanced_boolean", "column_read", "legacy only"),
            ("imbalanced_target", "column_read", "legacy only"),
            ("inconsistent_datetime", "column_read", "legacy only"),
            ("mixed_boolean_types", "column_read", "legacy only"),
            ("mostly_zeros", "column_read", "legacy only"),
            ("multiclass_target", "column_read", "legacy only"),
            ("near_zero_variance", "column_read", "legacy only"),
            ("normal_boolean", "column_read", "legacy only"),
            ("normal_categorical", "column_read", "legacy only"),
            ("normal_date", "column_read", "legacy only"),
            ("perfect_leakage", "column_read", "legacy only"),
            ("short_text", "column_read", "legacy only"),
            ("target_for_leakage", "column_read", "legacy only"),
        ],
    ),
    *_group(
        "sms_spam",
        SMS_PAIRS,
        [
            ("", "finding", "legacy info, core warn"),
            ("", "pairs_flagged", "legacy 5, core 217"),
        ],
    ),
    *_group(
        "sms_spam",
        LABELS,
        [
            ("label", "column_read", "legacy only"),
        ],
    ),
)
