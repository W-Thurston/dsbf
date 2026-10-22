"""What differs between the group difference and collinearity checks and their tasks.

See ``accepted.py`` for what an entry is. The rows are what ``test_migration_parity``
printed for the golden datasets, grouped by the reason each is right. The F, H, p, n
and significance of every pair that `one_way_anova` and `kruskal_wallis` tested match
the check's on all ten datasets, so nothing of theirs is listed but the datasets with
no pair to test.
"""

from __future__ import annotations

from collections.abc import Iterable

from tests.golden.migration.harness import Accepted

CENTRED = (
    "The task fitted each column on the others without an intercept, which "
    "`statsmodels` before 0.15 does, so a column with a large mean scores a high VIF "
    "from its mean alone: on `clean`, `age`, `income`, `score` and `tenure_years` are "
    "independent random draws and scored 33.5, 26.7, 24.6 and 11.3, where the VIF of "
    "a column the others do not predict is 1. `collinearity` centres the columns, "
    "which is the VIF as defined (1 / (1 - R²) of a regression with an intercept, "
    "the diagonal of the inverse correlation matrix, and what statsmodels 0.15 "
    "computes), and scores those four 1.000 to 1.001. Where columns are truly "
    "collinear the two agree to within a few percent; the uncentred value is far "
    "higher only where means add to the collinearity (`signal` and `sparse_c` of "
    "`severe_multicollinearity`: 917 and 908 against 36.5)."
)
CENTRED_FINDING = (
    "The legacy finding came from the uncentred VIF: the column's centred VIF is "
    "under 5, so the check, which reports a VIF from 5, raises none."
)
SEVERITY = (
    "The task called a VIF of 20 or more an error. A high VIF is a hazard for a "
    "linear model and not a defect in the data, so the check reports it as a warning "
    "at every height from 10; the VIF and its tier (`severe`) are in the title and "
    "evidence."
)
SPARSE = (
    "The task dropped every row with a missing value in any numeric column, so its "
    "VIF rested on the rows complete in all of them. `collinearity` leaves out a "
    "column more than 40% missing first (`max_missing_share`), says so in "
    "`skipped_columns`, and keeps the rows the other columns share (F-32)."
)
CODED = (
    "The profile types an integer column with few distinct values categorical (a "
    "code or a small count), so `collinearity`, which reads numeric columns, does "
    "not include it. A VIF of a code against measurements has no meaning."
)
SINGULAR = (
    "The legacy fit is singular on this frame: `constant_numeric` is in the matrix "
    "beside exact copies (`corr_base`, `highly_correlated`), and the uncentred "
    "regression of a column on a constant and copies is undetermined. Its values "
    "are below 1 (0.08 for `normal_numeric_float`; reproduced with statsmodels 0.14 "
    "on the 5 250 complete rows), which a VIF cannot be. The check leaves the "
    "constant out and scores independent columns 1.0 and the copies 900 and more."
)
SINGULAR_FOUND = (
    "The collinearity the legacy fit hid: `corr_base` and the columns generated "
    "from it are copies with small noise, and the check reports them (VIF 900 and "
    "more) where the legacy fit's values were below 1."
)
NOT_NUMERIC = (
    "These columns are constant, or codes the profile types categorical, so the "
    "check does not read them; the legacy matrix included them, which is why it was "
    "singular."
)
NO_NUMERIC_PAIR = (
    "With one column there is nothing to regress it on; the task returned an empty "
    "successful result and the check ends not applicable and says so."
)
NO_PAIR = (
    "There is no numeric column paired with a categorical one (no numeric column, "
    "or no categorical column with 2 to 20 levels), so the task returned an empty "
    "successful result and the check ends not applicable and says so."
)
CONTRASTS_TASK_ONLY = (
    "The task tested the first ten level pairs of every pair of columns, whether or "
    "not the groups differed, in order of first appearance, so which pairs were "
    "tested depended on the row order. The check contrasts every level pair of a "
    "pair its omnibus tests flag (up to 10 groups) and corrects them as one family, "
    "so a pair whose groups do not differ gets none. The task's significance is not "
    "compared: its Benjamini-Hochberg family was every test run, which grows with "
    "the number of levels."
)
CONTRASTS_CHECK_ONLY = (
    "The task stopped at the first ten level pairs of each column, and a column of "
    "five or more levels has more (`sibsp`, `parch` and `deck` on titanic have 15 "
    "or more pairs of levels). The check contrasts every level pair of a flagged "
    "pair, up to 10 groups."
)
LEVELS_MW = (
    "The task took the first ten level pairs of columns with at most ten levels, "
    "and `future_leakage` has more, so it never contrasted the pair that differs "
    "most. The check finds it through the omnibus tests, which read up to 20 levels."
)


def _group(
    check: str, dataset: str, reason: str, rows: Iterable[tuple[str, str, str]]
) -> list[Accepted]:
    return [Accepted(check, dataset, s, k, d, reason) for s, k, d in rows]


ACCEPTED_GROUP_DIFFERENCES: tuple[Accepted, ...] = (
    *_group(
        "detect_collinear_features",
        "clean",
        CENTRED,
        [
            ("age", "vif", "legacy 33.5, core 1"),
            ("income", "vif", "legacy 26.7, core 1"),
            ("score", "vif", "legacy 24.6, core 1"),
            ("tenure_years", "vif", "legacy 11.3, core 1"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "clean",
        CENTRED_FINDING,
        [
            ("age", "finding", "legacy only (error)"),
            ("income", "finding", "legacy only (error)"),
            ("score", "finding", "legacy only (error)"),
            ("tenure_years", "finding", "legacy only (warn)"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "high_missingness",
        CENTRED,
        [
            ("age", "vif", "legacy 14.8, core 1"),
            ("income", "vif", "legacy 9.05, core 1"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "high_missingness",
        CENTRED_FINDING,
        [
            ("age", "finding", "legacy only (warn)"),
            ("income", "finding", "legacy only (info)"),
            ("premium_score", "finding", "legacy only (info)"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "high_missingness",
        SPARSE,
        [
            ("premium_score", "vif", "legacy only"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "kitchen5k",
        NOT_NUMERIC,
        [
            ("balanced_target", "vif", "legacy only"),
            ("constant_numeric", "vif", "legacy only"),
            ("imbalanced_target", "vif", "legacy only"),
            ("mostly_zeros", "vif", "legacy only"),
            ("multiclass_target", "vif", "legacy only"),
            ("near_zero_variance", "vif", "legacy only"),
            ("perfect_leakage", "vif", "legacy only"),
            ("target_for_leakage", "vif", "legacy only"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "kitchen5k",
        SINGULAR,
        [
            ("age_with_bounds_issues", "vif", "legacy 0.501, core 1"),
            ("bimodal_column", "vif", "legacy 0.269, core 1"),
            ("corr_base", "vif", "legacy 0.0841, core 928"),
            ("duplicate_col_1", "vif", "legacy 0.244, core 1e+06"),
            ("duplicate_col_2", "vif", "legacy 0.244, core 1e+06"),
            ("duplicate_col_3", "vif", "legacy 0.244, core 1e+06"),
            ("highly_correlated", "vif", "legacy 0.0841, core 915"),
            ("left_skewed", "vif", "legacy 0.496, core 1"),
            ("moderately_correlated", "vif", "legacy 0.114, core 3.35"),
            ("negative_counts", "vif", "legacy 0.433, core 1"),
            ("negatively_correlated", "vif", "legacy 0.0913, core 10.1"),
            ("normal_numeric_float", "vif", "legacy 0.0823, core 1.01"),
            ("normal_numeric_int", "vif", "legacy 0.251, core 1"),
            ("outlier_column", "vif", "legacy 0.751, core 1"),
            ("partial_duplicate_1", "vif", "legacy 0.259, core 8.6e+03"),
            ("partial_duplicate_2", "vif", "legacy 0.259, core 8.6e+03"),
            ("percentage_out_of_bounds", "vif", "legacy 0.574, core 1.01"),
            ("right_skewed", "vif", "legacy 0.506, core 1"),
            ("sequential_id", "vif", "legacy 0.25, core 1"),
            ("sparse_column", "vif", "legacy 0.954, core 1"),
            ("subtle_leakage", "vif", "legacy 0.696, core 1"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "kitchen5k",
        SINGULAR_FOUND,
        [
            ("corr_base", "finding", "core only (warn)"),
            ("duplicate_col_1", "finding", "core only (warn)"),
            ("duplicate_col_2", "finding", "core only (warn)"),
            ("duplicate_col_3", "finding", "core only (warn)"),
            ("highly_correlated", "finding", "core only (warn)"),
            ("negatively_correlated", "finding", "core only (warn)"),
            ("partial_duplicate_1", "finding", "core only (warn)"),
            ("partial_duplicate_2", "finding", "core only (warn)"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "near_clean",
        CENTRED,
        [
            ("age", "vif", "legacy 24.4, core 1"),
            ("income", "vif", "legacy 14.1, core 1"),
            ("purchase_amount", "vif", "legacy 1.98, core 1"),
            ("satisfaction", "vif", "legacy 14.5, core 1"),
            ("tenure_months", "vif", "legacy 7.05, core 1"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "near_clean",
        CENTRED_FINDING,
        [
            ("age", "finding", "legacy only (error)"),
            ("income", "finding", "legacy only (warn)"),
            ("satisfaction", "finding", "legacy only (warn)"),
            ("tenure_months", "finding", "legacy only (info)"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "severe_multicollinearity",
        CENTRED,
        [
            ("base", "vif", "legacy 64.1, core 61.7"),
            ("click_rate", "vif", "legacy 7.78, core 1.01"),
            ("copy_a", "vif", "legacy 84, core 77.4"),
            ("copy_b", "vif", "legacy 16.4, core 16.8"),
            ("factor_x", "vif", "legacy 37, core 35.2"),
            ("factor_y", "vif", "legacy 23.6, core 20.1"),
            ("revenue", "vif", "legacy 5.99, core 1.01"),
            ("signal", "vif", "legacy 917, core 1"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "severe_multicollinearity",
        CENTRED_FINDING,
        [
            ("click_rate", "finding", "legacy only (info)"),
            ("revenue", "finding", "legacy only (info)"),
            ("signal", "finding", "legacy only (error)"),
            ("sparse_c", "finding", "legacy only (error)"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "severe_multicollinearity",
        SEVERITY,
        [
            ("base", "severity", "legacy error, core warn"),
            ("combo", "severity", "legacy error, core warn"),
            ("copy_a", "severity", "legacy error, core warn"),
            ("factor_x", "severity", "legacy error, core warn"),
            ("factor_y", "severity", "legacy error, core warn"),
            ("factor_z", "severity", "legacy error, core warn"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "severe_multicollinearity",
        SPARSE,
        [
            ("sparse_c", "vif", "legacy only"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "single_column",
        NO_NUMERIC_PAIR,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "tiny",
        CENTRED,
        [
            ("x", "vif", "legacy 10.4, core 1.11"),
            ("y", "vif", "legacy 21.8, core 1.38"),
            ("z", "vif", "legacy 14.3, core 1.28"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "tiny",
        CENTRED_FINDING,
        [
            ("x", "finding", "legacy only (warn)"),
            ("y", "finding", "legacy only (error)"),
            ("z", "finding", "legacy only (warn)"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "titanic",
        CENTRED,
        [
            ("age", "vif", "legacy 3.48, core 1.01"),
            ("fare", "vif", "legacy 1.84, core 1.01"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "titanic",
        CODED,
        [
            ("parch", "vif", "legacy only"),
            ("pclass", "vif", "legacy only"),
            ("sibsp", "vif", "legacy only"),
            ("survived", "vif", "legacy only"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "wide",
        CENTRED,
        [
            ("cont_clean_00", "vif", "legacy 9.34, core 1.18"),
            ("cont_clean_01", "vif", "legacy 72.5, core 1.23"),
            ("cont_clean_02", "vif", "legacy 1.27, core 1.21"),
            ("cont_clean_03", "vif", "legacy 1.71, core 1.27"),
            ("cont_clean_04", "vif", "legacy 14.4, core 1.19"),
            ("cont_clean_05", "vif", "legacy 52.4, core 1.19"),
            ("cont_clean_06", "vif", "legacy 13.5, core 1.24"),
            ("cont_clean_07", "vif", "legacy 14.9, core 1.16"),
            ("cont_clean_08", "vif", "legacy 43.8, core 1.23"),
            ("cont_clean_09", "vif", "legacy 7.37, core 1.17"),
            ("cont_clean_10", "vif", "legacy 17.6, core 1.23"),
            ("cont_clean_11", "vif", "legacy 1.72, core 1.19"),
            ("cont_clean_12", "vif", "legacy 1.97, core 1.21"),
            ("cont_clean_13", "vif", "legacy 1.35, core 1.18"),
            ("cont_clean_14", "vif", "legacy 5.67, core 1.19"),
            ("cont_clean_15", "vif", "legacy 2.13, core 1.21"),
            ("cont_clean_16", "vif", "legacy 17.7, core 1.16"),
            ("cont_clean_17", "vif", "legacy 1.38, core 1.2"),
            ("cont_clean_18", "vif", "legacy 59.5, core 1.16"),
            ("cont_clean_19", "vif", "legacy 89.6, core 1.19"),
            ("cont_clean_20", "vif", "legacy 53.4, core 1.17"),
            ("cont_clean_21", "vif", "legacy 18.3, core 1.22"),
            ("cont_clean_22", "vif", "legacy 35.1, core 1.26"),
            ("cont_clean_23", "vif", "legacy 21.3, core 1.21"),
            ("cont_clean_24", "vif", "legacy 3.04, core 1.2"),
            ("cont_clean_25", "vif", "legacy 18.7, core 1.27"),
            ("cont_clean_26", "vif", "legacy 24.3, core 1.24"),
            ("cont_clean_28", "vif", "legacy 1.84, core 1.19"),
            ("cont_clean_29", "vif", "legacy 5.17, core 1.22"),
            ("cont_clean_30", "vif", "legacy 33.2, core 1.18"),
            ("cont_clean_31", "vif", "legacy 201, core 1.29"),
            ("cont_clean_32", "vif", "legacy 6.54, core 1.21"),
            ("cont_clean_33", "vif", "legacy 144, core 1.21"),
            ("cont_clean_34", "vif", "legacy 158, core 1.21"),
            ("cont_clean_35", "vif", "legacy 15.4, core 1.21"),
            ("cont_clean_36", "vif", "legacy 1.2, core 1.17"),
            ("cont_clean_37", "vif", "legacy 1.65, core 1.16"),
            ("cont_clean_38", "vif", "legacy 1.66, core 1.22"),
            ("cont_clean_39", "vif", "legacy 140, core 1.25"),
            ("cont_skew_00", "vif", "legacy 3.81, core 1.18"),
            ("cont_skew_01", "vif", "legacy 3.65, core 1.21"),
            ("cont_skew_02", "vif", "legacy 3.24, core 1.15"),
            ("cont_skew_03", "vif", "legacy 4.24, core 1.21"),
            ("cont_skew_04", "vif", "legacy 4.08, core 1.19"),
            ("cont_skew_05", "vif", "legacy 3.43, core 1.25"),
            ("cont_skew_06", "vif", "legacy 3.46, core 1.23"),
            ("cont_skew_07", "vif", "legacy 3.6, core 1.17"),
            ("cont_skew_08", "vif", "legacy 3.64, core 1.17"),
            ("cont_skew_09", "vif", "legacy 3.86, core 1.22"),
            ("cont_skew_10", "vif", "legacy 3.36, core 1.19"),
            ("cont_skew_11", "vif", "legacy 3.5, core 1.21"),
            ("cont_skew_12", "vif", "legacy 3.38, core 1.22"),
            ("cont_skew_13", "vif", "legacy 3.31, core 1.19"),
            ("cont_skew_14", "vif", "legacy 3.49, core 1.17"),
            ("cont_skew_15", "vif", "legacy 3.46, core 1.17"),
            ("cont_skew_16", "vif", "legacy 3.59, core 1.22"),
            ("cont_skew_17", "vif", "legacy 3.74, core 1.2"),
            ("cont_skew_18", "vif", "legacy 3.61, core 1.19"),
            ("cont_skew_19", "vif", "legacy 4.01, core 1.23"),
            ("null_00", "vif", "legacy 13.9, core 1.17"),
            ("null_01", "vif", "legacy 15, core 1.17"),
            ("null_02", "vif", "legacy 15.3, core 1.2"),
            ("null_03", "vif", "legacy 13.5, core 1.19"),
            ("null_04", "vif", "legacy 15.7, core 1.23"),
            ("null_05", "vif", "legacy 12.9, core 1.15"),
            ("null_06", "vif", "legacy 13.9, core 1.18"),
            ("null_07", "vif", "legacy 14.6, core 1.23"),
        ],
    ),
    *_group(
        "detect_collinear_features",
        "wide",
        CENTRED_FINDING,
        [
            ("cont_clean_00", "finding", "legacy only (info)"),
            ("cont_clean_01", "finding", "legacy only (error)"),
            ("cont_clean_04", "finding", "legacy only (warn)"),
            ("cont_clean_05", "finding", "legacy only (error)"),
            ("cont_clean_06", "finding", "legacy only (warn)"),
            ("cont_clean_07", "finding", "legacy only (warn)"),
            ("cont_clean_08", "finding", "legacy only (error)"),
            ("cont_clean_09", "finding", "legacy only (info)"),
            ("cont_clean_10", "finding", "legacy only (warn)"),
            ("cont_clean_14", "finding", "legacy only (info)"),
            ("cont_clean_16", "finding", "legacy only (warn)"),
            ("cont_clean_18", "finding", "legacy only (error)"),
            ("cont_clean_19", "finding", "legacy only (error)"),
            ("cont_clean_20", "finding", "legacy only (error)"),
            ("cont_clean_21", "finding", "legacy only (warn)"),
            ("cont_clean_22", "finding", "legacy only (error)"),
            ("cont_clean_23", "finding", "legacy only (error)"),
            ("cont_clean_25", "finding", "legacy only (warn)"),
            ("cont_clean_26", "finding", "legacy only (error)"),
            ("cont_clean_29", "finding", "legacy only (info)"),
            ("cont_clean_30", "finding", "legacy only (error)"),
            ("cont_clean_31", "finding", "legacy only (error)"),
            ("cont_clean_32", "finding", "legacy only (info)"),
            ("cont_clean_33", "finding", "legacy only (error)"),
            ("cont_clean_34", "finding", "legacy only (error)"),
            ("cont_clean_35", "finding", "legacy only (warn)"),
            ("cont_clean_39", "finding", "legacy only (error)"),
            ("null_00", "finding", "legacy only (warn)"),
            ("null_01", "finding", "legacy only (warn)"),
            ("null_02", "finding", "legacy only (warn)"),
            ("null_03", "finding", "legacy only (warn)"),
            ("null_04", "finding", "legacy only (warn)"),
            ("null_05", "finding", "legacy only (warn)"),
            ("null_06", "finding", "legacy only (warn)"),
            ("null_07", "finding", "legacy only (warn)"),
        ],
    ),
    *_group(
        "kruskal_wallis",
        "all_categorical",
        NO_PAIR,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "kruskal_wallis",
        "severe_multicollinearity",
        NO_PAIR,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "kruskal_wallis",
        "single_column",
        NO_PAIR,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "mann_whitney_u",
        "all_categorical",
        NO_PAIR,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "mann_whitney_u",
        "clean",
        CONTRASTS_TASK_ONLY,
        [
            ("", "contrasts", "68 tested by the task only"),
        ],
    ),
    *_group(
        "mann_whitney_u",
        "high_missingness",
        CONTRASTS_TASK_ONLY,
        [
            ("", "contrasts", "87 tested by the task only"),
        ],
    ),
    *_group(
        "mann_whitney_u",
        "kitchen5k",
        CONTRASTS_TASK_ONLY,
        [
            ("", "contrasts", "1162 tested by the task only"),
        ],
    ),
    *_group(
        "mann_whitney_u",
        "kitchen5k",
        LEVELS_MW,
        [
            ("future_leakage", "finding", "core only (info)"),
        ],
    ),
    *_group(
        "mann_whitney_u",
        "near_clean",
        CONTRASTS_TASK_ONLY,
        [
            ("", "contrasts", "65 tested by the task only"),
        ],
    ),
    *_group(
        "mann_whitney_u",
        "severe_multicollinearity",
        NO_PAIR,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "mann_whitney_u",
        "single_column",
        NO_PAIR,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "mann_whitney_u",
        "tiny",
        CONTRASTS_TASK_ONLY,
        [
            ("", "contrasts", "12 tested by the task only"),
        ],
    ),
    *_group(
        "mann_whitney_u",
        "titanic",
        CONTRASTS_CHECK_ONLY,
        [
            ("", "contrasts", "37 run by the check only"),
        ],
    ),
    *_group(
        "mann_whitney_u",
        "titanic",
        CONTRASTS_TASK_ONLY,
        [
            ("", "contrasts", "9 tested by the task only"),
        ],
    ),
    *_group(
        "mann_whitney_u",
        "wide",
        CONTRASTS_TASK_ONLY,
        [
            ("", "contrasts", "10948 tested by the task only"),
        ],
    ),
    *_group(
        "one_way_anova",
        "all_categorical",
        NO_PAIR,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "one_way_anova",
        "severe_multicollinearity",
        NO_PAIR,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "one_way_anova",
        "single_column",
        NO_PAIR,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
)
