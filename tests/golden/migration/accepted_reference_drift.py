"""What differs between `reference_drift` and the three drift tasks.

See ``accepted.py`` for what an entry is. The first group is every dataset run without a
reference; the rest are the runs against the perturbed copy ``reference_one.py`` makes,
under ``<dataset>+reference``: some numeric columns moved by half a standard deviation,
one categorical column reweighted, nulls added to a numeric column and the last column
removed. The tasks ran on the Polars backend, which two of them need.
"""

from __future__ import annotations

from collections.abc import Iterable

from tests.golden.datasets import GOLDEN_DATASETS
from tests.golden.migration.harness import Accepted

NO_REFERENCE = (
    "No reference was given, which is the default. The task ended skipped with a "
    "message; the check ends not applicable (`no_reference`), and says so in the plan "
    "before the run."
)
WIDENED = (
    "A column of whole numbers became decimal numbers (the reference moves it by half "
    "a standard deviation, as a nullable integer column read with nulls would). "
    "The task called every change of dtype a type mismatch; the check compares the "
    "kind of storage, so integers and floats are both numbers and nothing is "
    "reported."
)
BOOLEAN = (
    "A boolean column. The task compared only numeric and string columns and listed "
    "booleans as unsupported; the check compares the shares of true and false."
)
STRINGS = (
    "A text, identifier, datetime or mixed-type column. The task compared every string "
    "column's values as categories, so an identifier with a value per row or a date "
    "with one per day came out as total drift from the reference whatever its "
    "distribution. The check compares numeric, boolean and categorical columns, and "
    "reports a changed type for the others."
)
BINS = (
    "A numeric column moved by half a standard deviation. The task's PSI cut the "
    "combined range into ten equal-width bins, in which the middle bins hold most of "
    "the rows and a shift of that size crosses at most one boundary, so it read low "
    "(PSI under 0.1, as 0.05 for titanic's age). The check merges the reference's "
    "histogram into at most ten bins that each hold a tenth of the reference, the "
    "bins PSI's limits were set for, and reads the same shift as 0.1 to 0.5."
)
HIGH_AT = (
    "The task called PSI of 0.2 or more high; the check calls it high from 0.25, the "
    "conventional limit for a major shift, and moderate from 0.1. This column sits "
    "between."
)
TAIL_BINS = (
    "A skewed column. Equal-width bins put nearly all of the rows in the first few, so "
    "the task's PSI moves little when the body of the distribution does; the check's "
    "bins hold a tenth of the reference each and see the shift."
)
CHI2_FLAG = (
    "The task rated the target's drift from the chi-squared p-value, which falls "
    "below any limit once there are some hundreds of rows: a class share that moved by "
    "a tenth (total variation distance 0.10) was `significant`. The check reads the "
    "size of the shift and calls it drift from 0.2."
)
CHI2_LEVEL = (
    "The target's class shares moved by more than 0.2 (total variation distance). The "
    "task called it `significant` because the chi-squared p-value fell under half its "
    "limit, as it does for any shift on this many rows; the check, which reads the "
    "size of the shift, calls 0.2 to 0.4 moderate (a warning, one level up for a "
    "target) and from 0.4 high."
)
CHI2_TOTALS = (
    "The task raised an error: it tests the target's class counts against the "
    "reference's with `chisquare`, which needs equal totals, and the two datasets have "
    "different numbers of non-null values. The check compares shares."
)
NO_TARGET = (
    "No column of this dataset has two to ten text values, so no target was set. The "
    "task skipped; the check, with no target to single out, compares every column the "
    "same."
)


def _group(
    task: str, dataset: str, reason: str, rows: Iterable[tuple[str, str, str]]
) -> list[Accepted]:
    return [Accepted(task, dataset, s, k, d, reason) for s, k, d in rows]


ACCEPTED_REFERENCE_DRIFT: tuple[Accepted, ...] = (
    *(
        accepted
        for task in (
            "compare_with_reference_dataset",
            "detect_feature_drift",
            "detect_target_drift",
        )
        for dataset in GOLDEN_DATASETS
        for accepted in _group(
            task,
            dataset.name,
            NO_REFERENCE,
            [("", "status", "legacy skipped, core not_applicable")],
        )
    ),
    *_group(
        "compare_with_reference_dataset",
        "clean+reference",
        WIDENED,
        [
            ("age", "type", "legacy only"),
        ],
    ),
    *_group(
        "compare_with_reference_dataset",
        "kitchen5k+reference",
        WIDENED,
        [
            ("duplicate_col_2", "type", "legacy only"),
            ("normal_numeric_int", "type", "legacy only"),
        ],
    ),
    *_group(
        "compare_with_reference_dataset",
        "near_clean+reference",
        WIDENED,
        [
            ("age", "type", "legacy only"),
            ("tenure_months", "type", "legacy only"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "all_categorical+reference",
        BOOLEAN,
        [
            ("is_active", "column_read", "core only"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "clean+reference",
        BINS,
        [
            ("age", "flagged", "core only (high)"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "clean+reference",
        BOOLEAN,
        [
            ("is_active", "column_read", "core only"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "high_missingness+reference",
        BINS,
        [
            ("age", "flagged", "core only (high)"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "kitchen5k+reference",
        BINS,
        [
            ("corr_base", "flagged", "core only (moderate)"),
            ("negatively_correlated", "flagged", "core only (moderate)"),
            ("percentage_out_of_bounds", "flagged", "core only (high)"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "kitchen5k+reference",
        BOOLEAN,
        [
            ("imbalanced_boolean", "column_read", "core only"),
            ("normal_boolean", "column_read", "core only"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "kitchen5k+reference",
        STRINGS,
        [
            ("base64_encoded", "column_read", "legacy only"),
            ("datetime_with_gaps", "column_read", "legacy only"),
            ("email_with_violations", "column_read", "legacy only"),
            ("hex_encoded", "column_read", "legacy only"),
            ("high_cardinality_uuid", "column_read", "legacy only"),
            ("long_text", "column_read", "legacy only"),
            ("mixed_type_column", "column_read", "legacy only"),
            ("normal_date", "column_read", "legacy only"),
            ("normal_text", "column_read", "legacy only"),
            ("phone_with_violations", "column_read", "legacy only"),
            ("transaction_id", "column_read", "legacy only"),
            ("user_id", "column_read", "legacy only"),
            ("uuid_column", "column_read", "legacy only"),
            ("variable_length_text", "column_read", "legacy only"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "kitchen5k+reference",
        TAIL_BINS,
        [
            ("partial_duplicate_2", "level", "legacy moderate, core high"),
            ("right_skewed", "level", "legacy moderate, core high"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "near_clean+reference",
        BINS,
        [
            ("age", "flagged", "core only (moderate)"),
            ("tenure_months", "flagged", "core only (high)"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "severe_multicollinearity+reference",
        BINS,
        [
            ("signal", "flagged", "core only (moderate)"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "severe_multicollinearity+reference",
        HIGH_AT,
        [
            ("base", "level", "legacy high, core moderate"),
            ("combo", "level", "legacy high, core moderate"),
            ("factor_x", "level", "legacy high, core moderate"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "single_column+reference",
        BINS,
        [
            ("value", "flagged", "core only (high)"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "tiny+reference",
        BOOLEAN,
        [
            ("flag", "column_read", "core only"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "titanic+reference",
        BINS,
        [
            ("age", "flagged", "core only (high)"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "titanic+reference",
        BOOLEAN,
        [
            ("adult_male", "column_read", "core only"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "wide+reference",
        BINS,
        [
            ("cont_clean_00", "flagged", "core only (high)"),
            ("cont_clean_03", "flagged", "core only (high)"),
            ("cont_clean_06", "flagged", "core only (moderate)"),
            ("cont_clean_09", "flagged", "core only (high)"),
            ("cont_clean_12", "flagged", "core only (moderate)"),
            ("cont_clean_15", "flagged", "core only (high)"),
            ("cont_clean_18", "flagged", "core only (moderate)"),
            ("cont_clean_21", "flagged", "core only (high)"),
            ("cont_clean_24", "flagged", "core only (moderate)"),
            ("cont_clean_27", "flagged", "core only (moderate)"),
            ("cont_clean_30", "flagged", "core only (high)"),
            ("cont_clean_33", "flagged", "core only (moderate)"),
            ("cont_clean_36", "flagged", "core only (moderate)"),
            ("cont_clean_39", "flagged", "core only (moderate)"),
            ("cont_skew_02", "flagged", "core only (high)"),
            ("cont_skew_05", "flagged", "core only (high)"),
            ("cont_skew_08", "flagged", "core only (high)"),
            ("cont_skew_11", "flagged", "core only (high)"),
            ("cont_skew_14", "flagged", "core only (high)"),
            ("null_00", "flagged", "core only (high)"),
            ("null_03", "flagged", "core only (moderate)"),
            ("null_06", "flagged", "core only (moderate)"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "wide+reference",
        BOOLEAN,
        [
            ("bool_00", "column_read", "core only"),
            ("bool_01", "column_read", "core only"),
            ("bool_02", "column_read", "core only"),
            ("bool_03", "column_read", "core only"),
            ("bool_04", "column_read", "core only"),
        ],
    ),
    *_group(
        "detect_feature_drift",
        "wide+reference",
        TAIL_BINS,
        [
            ("cont_skew_17", "level", "legacy moderate, core high"),
        ],
    ),
    *_group(
        "detect_target_drift",
        "all_categorical+reference",
        CHI2_LEVEL,
        [
            ("target", "level", "legacy significant, core warn"),
        ],
    ),
    *_group(
        "detect_target_drift",
        "clean+reference",
        CHI2_LEVEL,
        [
            ("target", "level", "legacy significant, core warn"),
        ],
    ),
    *_group(
        "detect_target_drift",
        "high_missingness+reference",
        CHI2_TOTALS,
        [
            ("", "status", "legacy error, core ok"),
        ],
    ),
    *_group(
        "detect_target_drift",
        "kitchen5k+reference",
        CHI2_LEVEL,
        [
            ("target", "level", "legacy significant, core warn"),
        ],
    ),
    *_group(
        "detect_target_drift",
        "near_clean+reference",
        CHI2_LEVEL,
        [
            ("target", "level", "legacy significant, core warn"),
        ],
    ),
    *_group(
        "detect_target_drift",
        "severe_multicollinearity+reference",
        NO_TARGET,
        [
            ("", "status", "legacy skipped, core ok"),
        ],
    ),
    *_group(
        "detect_target_drift",
        "single_column+reference",
        NO_TARGET,
        [
            ("", "status", "legacy skipped, core ok"),
        ],
    ),
    *_group(
        "detect_target_drift",
        "sms_spam+reference",
        CHI2_FLAG,
        [
            ("target", "flagged", "legacy significant, core none"),
        ],
    ),
    *_group(
        "detect_target_drift",
        "titanic+reference",
        CHI2_FLAG,
        [
            ("target", "flagged", "legacy significant, core none"),
        ],
    ),
    *_group(
        "detect_target_drift",
        "wide+reference",
        CHI2_LEVEL,
        [
            ("target", "level", "legacy significant, core warn"),
        ],
    ),
)
