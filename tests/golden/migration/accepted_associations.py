"""What differs between the pairwise checks and their tasks, accepted.

See ``accepted.py`` for what an entry is. The rows are what ``test_migration_parity``
printed for the golden datasets, grouped by the reason each is right.
"""

from __future__ import annotations

from collections.abc import Iterable

from tests.golden.migration.harness import Accepted

YATES = (
    "scipy's `chi2_contingency` applies Yates's continuity correction to a 2x2 table "
    "by default, which lowers chi-squared and V and raises the p-value (titanic's "
    "`sex` by `alive`: chi-squared 260.7 with it, 263.1 without, V 0.5409 against "
    "0.5434). The check reports Pearson's chi-squared and plain V, which are what "
    "their definitions give and what `cramers_v_corrected` starts from; the "
    "correction for small samples is Bergsma's, applied to V and not to the test."
)
LEVELS = (
    "The task limited the levels of a categorical column to 50 only for pairs of "
    "categoricals, so it computed η² of every numeric column against a column of "
    "hundreds or thousands of levels. Each group then holds a few rows, and η² of "
    "that many groups measures the groups' own noise: it approaches 1 as groups "
    "approach one row. `associations` skips a categorical column over `max_levels` "
    "(50) and says so in `skipped_columns`."
)
GATE = (
    "The task skipped a pair with fewer than 30 complete rows before it routed to a "
    "statistic, so its small-sample Kendall's tau was unreachable and a 30-row frame "
    "got no pairs. The check gates at `min_n` (8) and routes after it, so these "
    "pairs get Kendall's tau-b."
)
TRUNCATED = (
    "The task cut each table to the ten most frequent levels of each column before "
    "testing it, so its chi-squared, V and p-value describe a different table when "
    "a column has more levels. The check tests the whole table (up to `max_levels` "
    "levels) and prints only the top levels, so the statistics are not compared."
)
TRUNC_SKIP = (
    "The ten most frequent levels of `future_leakage` all belong to one class of the "
    "other column (the generator plants an outcome per class), so the task's cut "
    "table had a single row and it skipped the pair. The whole table has two rows "
    "and a corrected V of 1, which the check reports."
)
CORRECTED = (
    "The pair's plain V (0.111) is just over the weak band and its bias-corrected V "
    "(0.096) just under it, so the task raised a weak-association finding and the "
    "check, which reports from a corrected V of 0.1, does not. The correction is "
    "right at n = 1000 with a 4x2 table: plain V is positively biased."
)
CODED = (
    "The task correlated every numeric dtype, including columns the profile types "
    "categorical (`perfect_leakage` and `target_for_leakage` are 0/1 integer codes, "
    "and the task correlated the codes with `subtle_leakage`). `associations` "
    "measures such a pair as η, a numeric column against a categorical one, and "
    "reports it as a strong association; `redundant_columns` is for two numeric "
    "columns that carry the same information."
)
EMPTY_RESULT = (
    "The task returned an empty successful result where there was nothing to pair "
    "(one column, or no numeric columns); the check ends not applicable and says so."
)
ONE_CATEGORICAL = (
    "The dataset has one column a pair statistic reads (`label`, two-valued); the "
    "other is free text, which none of them reads. The task returned an empty "
    "successful result; the check ends not applicable and says so."
)


def _group(
    check: str, dataset: str, reason: str, rows: Iterable[tuple[str, str, str]]
) -> list[Accepted]:
    return [Accepted(check, dataset, s, k, d, reason) for s, k, d in rows]


ACCEPTED_ASSOCIATIONS: tuple[Accepted, ...] = (
    *_group(
        "compute_pairwise_associations",
        "all_categorical",
        YATES,
        [
            ("dominant|is_active", "cramers_v", "legacy 0.014073, core 0.0169333"),
            ("dominant|is_premium", "cramers_v", "legacy 0.005715, core 0.00896253"),
            ("is_active|is_premium", "cramers_v", "legacy 0.001953, core 0.00312156"),
        ],
    ),
    *_group(
        "compute_pairwise_associations",
        "clean",
        YATES,
        [
            ("is_active|has_upgrade", "cramers_v", "legacy 0.010542, core 0.0115489"),
        ],
    ),
    *_group(
        "compute_pairwise_associations",
        "kitchen5k",
        LEVELS,
        [
            ("age_with_bounds_issues|unexpected_object_column", "pair", "legacy only"),
            ("bimodal_column|extremely_long_strings", "pair", "legacy only"),
            ("bimodal_column|unexpected_object_column", "pair", "legacy only"),
            ("corr_base|unexpected_object_column", "pair", "legacy only"),
            ("duplicate_col_1|extremely_long_strings", "pair", "legacy only"),
            ("duplicate_col_1|high_cardinality_random", "pair", "legacy only"),
            ("duplicate_col_1|unexpected_object_column", "pair", "legacy only"),
            ("duplicate_col_2|extremely_long_strings", "pair", "legacy only"),
            ("duplicate_col_2|high_cardinality_random", "pair", "legacy only"),
            ("duplicate_col_2|unexpected_object_column", "pair", "legacy only"),
            ("duplicate_col_3|extremely_long_strings", "pair", "legacy only"),
            ("duplicate_col_3|high_cardinality_random", "pair", "legacy only"),
            ("duplicate_col_3|unexpected_object_column", "pair", "legacy only"),
            ("extreme_skew|extremely_long_strings", "pair", "legacy only"),
            ("extreme_skew|unexpected_object_column", "pair", "legacy only"),
            ("extremely_long_strings|age_with_bounds_issues", "pair", "legacy only"),
            ("extremely_long_strings|corr_base", "pair", "legacy only"),
            ("extremely_long_strings|highly_correlated", "pair", "legacy only"),
            ("extremely_long_strings|moderately_correlated", "pair", "legacy only"),
            ("extremely_long_strings|negative_counts", "pair", "legacy only"),
            ("extremely_long_strings|negatively_correlated", "pair", "legacy only"),
            ("extremely_long_strings|percentage_out_of_bounds", "pair", "legacy only"),
            ("extremely_long_strings|subtle_leakage", "pair", "legacy only"),
            ("high_cardinality_random|age_with_bounds_issues", "pair", "legacy only"),
            ("high_cardinality_random|bimodal_column", "pair", "legacy only"),
            ("high_cardinality_random|corr_base", "pair", "legacy only"),
            ("high_cardinality_random|extreme_skew", "pair", "legacy only"),
            ("high_cardinality_random|highly_correlated", "pair", "legacy only"),
            ("high_cardinality_random|left_skewed", "pair", "legacy only"),
            ("high_cardinality_random|moderately_correlated", "pair", "legacy only"),
            ("high_cardinality_random|negative_counts", "pair", "legacy only"),
            ("high_cardinality_random|negatively_correlated", "pair", "legacy only"),
            ("high_cardinality_random|outlier_column", "pair", "legacy only"),
            ("high_cardinality_random|percentage_out_of_bounds", "pair", "legacy only"),
            ("high_cardinality_random|right_skewed", "pair", "legacy only"),
            ("high_cardinality_random|sequential_id", "pair", "legacy only"),
            ("high_cardinality_random|sparse_column", "pair", "legacy only"),
            ("high_cardinality_random|subtle_leakage", "pair", "legacy only"),
            ("highly_correlated|unexpected_object_column", "pair", "legacy only"),
            ("left_skewed|extremely_long_strings", "pair", "legacy only"),
            ("left_skewed|unexpected_object_column", "pair", "legacy only"),
            ("moderately_correlated|unexpected_object_column", "pair", "legacy only"),
            ("negative_counts|unexpected_object_column", "pair", "legacy only"),
            ("negatively_correlated|unexpected_object_column", "pair", "legacy only"),
            ("normal_numeric_float|extremely_long_strings", "pair", "legacy only"),
            ("normal_numeric_float|high_cardinality_random", "pair", "legacy only"),
            ("normal_numeric_float|unexpected_object_column", "pair", "legacy only"),
            ("normal_numeric_int|extremely_long_strings", "pair", "legacy only"),
            ("normal_numeric_int|high_cardinality_random", "pair", "legacy only"),
            ("normal_numeric_int|unexpected_object_column", "pair", "legacy only"),
            ("outlier_column|extremely_long_strings", "pair", "legacy only"),
            ("outlier_column|unexpected_object_column", "pair", "legacy only"),
            ("partial_duplicate_1|extremely_long_strings", "pair", "legacy only"),
            ("partial_duplicate_1|high_cardinality_random", "pair", "legacy only"),
            ("partial_duplicate_1|unexpected_object_column", "pair", "legacy only"),
            ("partial_duplicate_2|extremely_long_strings", "pair", "legacy only"),
            ("partial_duplicate_2|high_cardinality_random", "pair", "legacy only"),
            ("partial_duplicate_2|unexpected_object_column", "pair", "legacy only"),
            (
                "percentage_out_of_bounds|unexpected_object_column",
                "pair",
                "legacy only",
            ),
            ("right_skewed|extremely_long_strings", "pair", "legacy only"),
            ("right_skewed|unexpected_object_column", "pair", "legacy only"),
            ("sequential_id|extremely_long_strings", "pair", "legacy only"),
            ("sequential_id|unexpected_object_column", "pair", "legacy only"),
            ("sparse_column|extremely_long_strings", "pair", "legacy only"),
            ("sparse_column|unexpected_object_column", "pair", "legacy only"),
            ("subtle_leakage|unexpected_object_column", "pair", "legacy only"),
        ],
    ),
    *_group(
        "compute_pairwise_associations",
        "kitchen5k",
        YATES,
        [
            (
                "balanced_target|imbalanced_target",
                "cramers_v",
                "legacy 0.024334, core 0.0252422",
            ),
            (
                "balanced_target|perfect_leakage",
                "cramers_v",
                "legacy 0.030281, core 0.0306941",
            ),
            (
                "balanced_target|target_for_leakage",
                "cramers_v",
                "legacy 0.030281, core 0.0306941",
            ),
            (
                "control_character_text|balanced_target",
                "cramers_v",
                "legacy 0.010869, core 0.0122358",
            ),
            (
                "control_character_text|imbalanced_boolean",
                "cramers_v",
                "legacy 0, core 0.00070101",
            ),
            (
                "control_character_text|imbalanced_target",
                "cramers_v",
                "legacy 0.001916, core 0.00517531",
            ),
            (
                "control_character_text|perfect_leakage",
                "cramers_v",
                "legacy 0.001107, core 0.00258997",
            ),
            (
                "control_character_text|target_for_leakage",
                "cramers_v",
                "legacy 0.001107, core 0.00258997",
            ),
            (
                "imbalanced_boolean|balanced_target",
                "cramers_v",
                "legacy 0.000565, core 0.00144616",
            ),
            (
                "imbalanced_boolean|imbalanced_target",
                "cramers_v",
                "legacy 0.014256, core 0.0163572",
            ),
            (
                "imbalanced_boolean|perfect_leakage",
                "cramers_v",
                "legacy 0.025938, core 0.0268933",
            ),
            (
                "imbalanced_boolean|target_for_leakage",
                "cramers_v",
                "legacy 0.025938, core 0.0268933",
            ),
            (
                "imbalanced_target|perfect_leakage",
                "cramers_v",
                "legacy 0.011166, core 0.0121513",
            ),
            (
                "imbalanced_target|target_for_leakage",
                "cramers_v",
                "legacy 0.011166, core 0.0121513",
            ),
            (
                "mostly_zeros|balanced_target",
                "cramers_v",
                "legacy 0.013896, core 0.014536",
            ),
            (
                "mostly_zeros|control_character_text",
                "cramers_v",
                "legacy 0, core 0.00101803",
            ),
            (
                "mostly_zeros|imbalanced_boolean",
                "cramers_v",
                "legacy 0.687112, core 0.688592",
            ),
            (
                "mostly_zeros|imbalanced_target",
                "cramers_v",
                "legacy 0.005232, core 0.00675812",
            ),
            (
                "mostly_zeros|perfect_leakage",
                "cramers_v",
                "legacy 0.00783, core 0.00852357",
            ),
            (
                "mostly_zeros|target_for_leakage",
                "cramers_v",
                "legacy 0.00783, core 0.00852357",
            ),
            (
                "normal_boolean|balanced_target",
                "cramers_v",
                "legacy 0.022034, core 0.022415",
            ),
            (
                "normal_boolean|control_character_text",
                "cramers_v",
                "legacy 0.026947, core 0.0283142",
            ),
            (
                "normal_boolean|imbalanced_boolean",
                "cramers_v",
                "legacy 0.009896, core 0.0107775",
            ),
            (
                "normal_boolean|imbalanced_target",
                "cramers_v",
                "legacy 0.009683, core 0.0105913",
            ),
            (
                "normal_boolean|mostly_zeros",
                "cramers_v",
                "legacy 0.004774, core 0.00541414",
            ),
            (
                "normal_boolean|perfect_leakage",
                "cramers_v",
                "legacy 0.015273, core 0.0156865",
            ),
            (
                "normal_boolean|target_for_leakage",
                "cramers_v",
                "legacy 0.015273, core 0.0156865",
            ),
            (
                "target_for_leakage|perfect_leakage",
                "cramers_v",
                "legacy 0.999552, core 1",
            ),
        ],
    ),
    *_group(
        "compute_pairwise_associations",
        "single_column",
        EMPTY_RESULT,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "compute_pairwise_associations",
        "tiny",
        GATE,
        [
            ("category|flag", "pair", "core only"),
            ("category|x", "pair", "core only"),
            ("category|y", "pair", "core only"),
            ("category|z", "pair", "core only"),
            ("flag|x", "pair", "core only"),
            ("flag|y", "pair", "core only"),
            ("flag|z", "pair", "core only"),
            ("x|y", "pair", "core only"),
            ("x|z", "pair", "core only"),
            ("y|z", "pair", "core only"),
        ],
    ),
    *_group(
        "compute_pairwise_associations",
        "titanic",
        YATES,
        [
            ("adult_male|alive", "cramers_v", "legacy 0.554722, core 0.55708"),
            ("adult_male|alone", "cramers_v", "legacy 0.4024, core 0.404744"),
            ("alive|alone", "cramers_v", "legacy 0.201009, core 0.203367"),
            ("sex|adult_male", "cramers_v", "legacy 0.906178, core 0.908578"),
            ("sex|alive", "cramers_v", "legacy 0.540936, core 0.543351"),
            ("sex|alone", "cramers_v", "legacy 0.301246, core 0.303646"),
            ("survived|adult_male", "cramers_v", "legacy 0.554722, core 0.55708"),
            ("survived|alive", "cramers_v", "legacy 0.997627, core 1"),
            ("survived|alone", "cramers_v", "legacy 0.201009, core 0.203367"),
            ("survived|sex", "cramers_v", "legacy 0.540936, core 0.543351"),
        ],
    ),
    *_group(
        "compute_pairwise_associations",
        "wide",
        LEVELS,
        [
            ("cat_hc|null_00", "pair", "legacy only"),
            ("cat_hc|null_01", "pair", "legacy only"),
            ("cat_hc|null_02", "pair", "legacy only"),
            ("cat_hc|null_03", "pair", "legacy only"),
            ("cat_hc|null_04", "pair", "legacy only"),
            ("cat_hc|null_05", "pair", "legacy only"),
            ("cat_hc|null_06", "pair", "legacy only"),
            ("cat_hc|null_07", "pair", "legacy only"),
            ("cont_clean_00|cat_hc", "pair", "legacy only"),
            ("cont_clean_01|cat_hc", "pair", "legacy only"),
            ("cont_clean_02|cat_hc", "pair", "legacy only"),
            ("cont_clean_03|cat_hc", "pair", "legacy only"),
            ("cont_clean_04|cat_hc", "pair", "legacy only"),
            ("cont_clean_05|cat_hc", "pair", "legacy only"),
            ("cont_clean_06|cat_hc", "pair", "legacy only"),
            ("cont_clean_07|cat_hc", "pair", "legacy only"),
            ("cont_clean_08|cat_hc", "pair", "legacy only"),
            ("cont_clean_09|cat_hc", "pair", "legacy only"),
            ("cont_clean_10|cat_hc", "pair", "legacy only"),
            ("cont_clean_11|cat_hc", "pair", "legacy only"),
            ("cont_clean_12|cat_hc", "pair", "legacy only"),
            ("cont_clean_13|cat_hc", "pair", "legacy only"),
            ("cont_clean_14|cat_hc", "pair", "legacy only"),
            ("cont_clean_15|cat_hc", "pair", "legacy only"),
            ("cont_clean_16|cat_hc", "pair", "legacy only"),
            ("cont_clean_17|cat_hc", "pair", "legacy only"),
            ("cont_clean_18|cat_hc", "pair", "legacy only"),
            ("cont_clean_19|cat_hc", "pair", "legacy only"),
            ("cont_clean_20|cat_hc", "pair", "legacy only"),
            ("cont_clean_21|cat_hc", "pair", "legacy only"),
            ("cont_clean_22|cat_hc", "pair", "legacy only"),
            ("cont_clean_23|cat_hc", "pair", "legacy only"),
            ("cont_clean_24|cat_hc", "pair", "legacy only"),
            ("cont_clean_25|cat_hc", "pair", "legacy only"),
            ("cont_clean_26|cat_hc", "pair", "legacy only"),
            ("cont_clean_27|cat_hc", "pair", "legacy only"),
            ("cont_clean_28|cat_hc", "pair", "legacy only"),
            ("cont_clean_29|cat_hc", "pair", "legacy only"),
            ("cont_clean_30|cat_hc", "pair", "legacy only"),
            ("cont_clean_31|cat_hc", "pair", "legacy only"),
            ("cont_clean_32|cat_hc", "pair", "legacy only"),
            ("cont_clean_33|cat_hc", "pair", "legacy only"),
            ("cont_clean_34|cat_hc", "pair", "legacy only"),
            ("cont_clean_35|cat_hc", "pair", "legacy only"),
            ("cont_clean_36|cat_hc", "pair", "legacy only"),
            ("cont_clean_37|cat_hc", "pair", "legacy only"),
            ("cont_clean_38|cat_hc", "pair", "legacy only"),
            ("cont_clean_39|cat_hc", "pair", "legacy only"),
            ("cont_skew_00|cat_hc", "pair", "legacy only"),
            ("cont_skew_01|cat_hc", "pair", "legacy only"),
            ("cont_skew_02|cat_hc", "pair", "legacy only"),
            ("cont_skew_03|cat_hc", "pair", "legacy only"),
            ("cont_skew_04|cat_hc", "pair", "legacy only"),
            ("cont_skew_05|cat_hc", "pair", "legacy only"),
            ("cont_skew_06|cat_hc", "pair", "legacy only"),
            ("cont_skew_07|cat_hc", "pair", "legacy only"),
            ("cont_skew_08|cat_hc", "pair", "legacy only"),
            ("cont_skew_09|cat_hc", "pair", "legacy only"),
            ("cont_skew_10|cat_hc", "pair", "legacy only"),
            ("cont_skew_11|cat_hc", "pair", "legacy only"),
            ("cont_skew_12|cat_hc", "pair", "legacy only"),
            ("cont_skew_13|cat_hc", "pair", "legacy only"),
            ("cont_skew_14|cat_hc", "pair", "legacy only"),
            ("cont_skew_15|cat_hc", "pair", "legacy only"),
            ("cont_skew_16|cat_hc", "pair", "legacy only"),
            ("cont_skew_17|cat_hc", "pair", "legacy only"),
            ("cont_skew_18|cat_hc", "pair", "legacy only"),
            ("cont_skew_19|cat_hc", "pair", "legacy only"),
        ],
    ),
    *_group(
        "compute_pairwise_associations",
        "wide",
        YATES,
        [
            ("bool_00|bool_01", "cramers_v", "legacy 0.020838, core 0.0229327"),
            ("bool_00|bool_02", "cramers_v", "legacy 0.000894, core 0.00308621"),
            ("bool_00|bool_03", "cramers_v", "legacy 0.02477, core 0.0268548"),
            ("bool_00|bool_04", "cramers_v", "legacy 0.017926, core 0.0201608"),
            ("bool_01|bool_02", "cramers_v", "legacy 0.017325, core 0.0194931"),
            ("bool_01|bool_03", "cramers_v", "legacy 0.00231, core 0.00437162"),
            ("bool_01|bool_04", "cramers_v", "legacy 0.014877, core 0.0170877"),
            ("bool_02|bool_03", "cramers_v", "legacy 0.037718, core 0.039876"),
            ("bool_02|bool_04", "cramers_v", "legacy 0, core 0.000185053"),
            ("bool_03|bool_04", "cramers_v", "legacy 0.015842, core 0.0180428"),
        ],
    ),
    *_group(
        "contingency_tables",
        "all_categorical",
        YATES,
        [
            ("dominant|is_active", "chi2", "legacy 0.3961, core 0.573473"),
            ("dominant|is_active", "cramers_v", "legacy 0.0141, core 0.0169333"),
            ("dominant|is_active", "p_value", "legacy 0.529113, core 0.448882"),
            ("dominant|is_premium", "chi2", "legacy 0.0653, core 0.160654"),
            ("dominant|is_premium", "cramers_v", "legacy 0.0057, core 0.00896253"),
            ("dominant|is_premium", "p_value", "legacy 0.798265, core 0.688555"),
            ("is_active|is_premium", "chi2", "legacy 0.0076, core 0.0194883"),
            ("is_active|is_premium", "cramers_v", "legacy 0.002, core 0.00312156"),
            ("is_active|is_premium", "p_value", "legacy 0.930389, core 0.888976"),
        ],
    ),
    *_group(
        "contingency_tables",
        "clean",
        YATES,
        [
            ("is_active|has_upgrade", "chi2", "legacy 0.2223, core 0.266755"),
            ("is_active|has_upgrade", "cramers_v", "legacy 0.0105, core 0.0115489"),
            ("is_active|has_upgrade", "p_value", "legacy 0.637305, core 0.605517"),
        ],
    ),
    *_group(
        "contingency_tables",
        "kitchen5k",
        TRUNCATED,
        [
            (
                "ambiguous_booleans|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "balanced_target|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "control_character_text|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "imbalanced_boolean|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "imbalanced_target|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "inconsistent_datetime|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "mixed_boolean_types|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "mostly_zeros|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "multiclass_target|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "near_zero_variance|ambiguous_booleans",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "near_zero_variance|balanced_target",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "near_zero_variance|control_character_text",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "near_zero_variance|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "near_zero_variance|imbalanced_boolean",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "near_zero_variance|imbalanced_target",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "near_zero_variance|inconsistent_datetime",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "near_zero_variance|mixed_boolean_types",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "near_zero_variance|mostly_zeros",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "near_zero_variance|multiclass_target",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "near_zero_variance|perfect_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "near_zero_variance|target_for_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "normal_boolean|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "normal_boolean|near_zero_variance",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "normal_categorical|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "normal_categorical|near_zero_variance",
                "truncated_table",
                "legacy tested the top levels",
            ),
            (
                "short_text|future_leakage",
                "truncated_table",
                "legacy tested the top levels",
            ),
        ],
    ),
    *_group(
        "contingency_tables",
        "kitchen5k",
        TRUNC_SKIP,
        [
            ("future_leakage", "finding", "core only (warn)"),
            ("future_leakage|perfect_leakage", "pair", "core only"),
            ("future_leakage|target_for_leakage", "pair", "core only"),
        ],
    ),
    *_group(
        "contingency_tables",
        "kitchen5k",
        YATES,
        [
            (
                "balanced_target|imbalanced_target",
                "chi2",
                "legacy 3.1087, core 3.34513",
            ),
            (
                "balanced_target|imbalanced_target",
                "cramers_v",
                "legacy 0.0243, core 0.0252422",
            ),
            (
                "balanced_target|imbalanced_target",
                "p_value",
                "legacy 0.077875, core 0.0674043",
            ),
            ("balanced_target|perfect_leakage", "chi2", "legacy 4.8139, core 4.94618"),
            (
                "balanced_target|perfect_leakage",
                "cramers_v",
                "legacy 0.0303, core 0.0306941",
            ),
            (
                "balanced_target|perfect_leakage",
                "p_value",
                "legacy 0.028231, core 0.0261484",
            ),
            (
                "balanced_target|target_for_leakage",
                "chi2",
                "legacy 4.8139, core 4.94618",
            ),
            (
                "balanced_target|target_for_leakage",
                "cramers_v",
                "legacy 0.0303, core 0.0306941",
            ),
            (
                "balanced_target|target_for_leakage",
                "p_value",
                "legacy 0.028231, core 0.0261484",
            ),
            (
                "control_character_text|balanced_target",
                "chi2",
                "legacy 0.6202, core 0.785999",
            ),
            (
                "control_character_text|balanced_target",
                "cramers_v",
                "legacy 0.0109, core 0.0122358",
            ),
            (
                "control_character_text|balanced_target",
                "p_value",
                "legacy 0.430979, core 0.375313",
            ),
            (
                "control_character_text|imbalanced_boolean",
                "chi2",
                "legacy 0, core 0.00257993",
            ),
            (
                "control_character_text|imbalanced_boolean",
                "cramers_v",
                "legacy 0, core 0.00070101",
            ),
            (
                "control_character_text|imbalanced_boolean",
                "p_value",
                "legacy 1, core 0.95949",
            ),
            (
                "control_character_text|imbalanced_target",
                "chi2",
                "legacy 0.0193, core 0.140615",
            ),
            (
                "control_character_text|imbalanced_target",
                "cramers_v",
                "legacy 0.0019, core 0.00517531",
            ),
            (
                "control_character_text|imbalanced_target",
                "p_value",
                "legacy 0.889592, core 0.70767",
            ),
            (
                "control_character_text|perfect_leakage",
                "chi2",
                "legacy 0.0064, core 0.0352166",
            ),
            (
                "control_character_text|perfect_leakage",
                "cramers_v",
                "legacy 0.0011, core 0.00258997",
            ),
            (
                "control_character_text|perfect_leakage",
                "p_value",
                "legacy 0.936042, core 0.851142",
            ),
            (
                "control_character_text|target_for_leakage",
                "chi2",
                "legacy 0.0064, core 0.0352166",
            ),
            (
                "control_character_text|target_for_leakage",
                "cramers_v",
                "legacy 0.0011, core 0.00258997",
            ),
            (
                "control_character_text|target_for_leakage",
                "p_value",
                "legacy 0.936042, core 0.851142",
            ),
            (
                "imbalanced_boolean|balanced_target",
                "chi2",
                "legacy 0.0017, core 0.0109798",
            ),
            (
                "imbalanced_boolean|balanced_target",
                "cramers_v",
                "legacy 0.0006, core 0.00144616",
            ),
            (
                "imbalanced_boolean|balanced_target",
                "p_value",
                "legacy 0.967347, core 0.916547",
            ),
            (
                "imbalanced_boolean|imbalanced_target",
                "chi2",
                "legacy 1.067, core 1.40468",
            ),
            (
                "imbalanced_boolean|imbalanced_target",
                "cramers_v",
                "legacy 0.0143, core 0.0163572",
            ),
            (
                "imbalanced_boolean|imbalanced_target",
                "p_value",
                "legacy 0.301625, core 0.235942",
            ),
            (
                "imbalanced_boolean|perfect_leakage",
                "chi2",
                "legacy 3.532, core 3.79706",
            ),
            (
                "imbalanced_boolean|perfect_leakage",
                "cramers_v",
                "legacy 0.0259, core 0.0268933",
            ),
            (
                "imbalanced_boolean|perfect_leakage",
                "p_value",
                "legacy 0.060195, core 0.0513425",
            ),
            (
                "imbalanced_boolean|target_for_leakage",
                "chi2",
                "legacy 3.532, core 3.79706",
            ),
            (
                "imbalanced_boolean|target_for_leakage",
                "cramers_v",
                "legacy 0.0259, core 0.0268933",
            ),
            (
                "imbalanced_boolean|target_for_leakage",
                "p_value",
                "legacy 0.060195, core 0.0513425",
            ),
            (
                "imbalanced_target|perfect_leakage",
                "chi2",
                "legacy 0.6546, core 0.775181",
            ),
            (
                "imbalanced_target|perfect_leakage",
                "cramers_v",
                "legacy 0.0112, core 0.0121513",
            ),
            (
                "imbalanced_target|perfect_leakage",
                "p_value",
                "legacy 0.418479, core 0.378619",
            ),
            (
                "imbalanced_target|target_for_leakage",
                "chi2",
                "legacy 0.6546, core 0.775181",
            ),
            (
                "imbalanced_target|target_for_leakage",
                "cramers_v",
                "legacy 0.0112, core 0.0121513",
            ),
            (
                "imbalanced_target|target_for_leakage",
                "p_value",
                "legacy 0.418479, core 0.378619",
            ),
            ("mostly_zeros|balanced_target", "chi2", "legacy 1.0138, core 1.10929"),
            (
                "mostly_zeros|balanced_target",
                "cramers_v",
                "legacy 0.0139, core 0.014536",
            ),
            (
                "mostly_zeros|balanced_target",
                "p_value",
                "legacy 0.313998, core 0.292235",
            ),
            (
                "mostly_zeros|control_character_text",
                "chi2",
                "legacy 0, core 0.00544107",
            ),
            (
                "mostly_zeros|control_character_text",
                "cramers_v",
                "legacy 0, core 0.00101803",
            ),
            (
                "mostly_zeros|control_character_text",
                "p_value",
                "legacy 1, core 0.941198",
            ),
            ("mostly_zeros|imbalanced_boolean", "chi2", "legacy 2478.64, core 2489.33"),
            (
                "mostly_zeros|imbalanced_boolean",
                "cramers_v",
                "legacy 0.6871, core 0.688592",
            ),
            ("mostly_zeros|imbalanced_target", "chi2", "legacy 0.1437, core 0.239779"),
            (
                "mostly_zeros|imbalanced_target",
                "cramers_v",
                "legacy 0.0052, core 0.00675812",
            ),
            (
                "mostly_zeros|imbalanced_target",
                "p_value",
                "legacy 0.704591, core 0.624366",
            ),
            ("mostly_zeros|perfect_leakage", "chi2", "legacy 0.3218, core 0.381419"),
            (
                "mostly_zeros|perfect_leakage",
                "cramers_v",
                "legacy 0.0078, core 0.00852357",
            ),
            (
                "mostly_zeros|perfect_leakage",
                "p_value",
                "legacy 0.570502, core 0.536845",
            ),
            ("mostly_zeros|target_for_leakage", "chi2", "legacy 0.3218, core 0.381419"),
            (
                "mostly_zeros|target_for_leakage",
                "cramers_v",
                "legacy 0.0078, core 0.00852357",
            ),
            (
                "mostly_zeros|target_for_leakage",
                "p_value",
                "legacy 0.570502, core 0.536845",
            ),
            ("normal_boolean|balanced_target", "chi2", "legacy 2.5489, core 2.63776"),
            (
                "normal_boolean|balanced_target",
                "cramers_v",
                "legacy 0.022, core 0.022415",
            ),
            (
                "normal_boolean|balanced_target",
                "p_value",
                "legacy 0.110374, core 0.104351",
            ),
            (
                "normal_boolean|control_character_text",
                "chi2",
                "legacy 3.8123, core 4.2089",
            ),
            (
                "normal_boolean|control_character_text",
                "cramers_v",
                "legacy 0.0269, core 0.0283142",
            ),
            (
                "normal_boolean|control_character_text",
                "p_value",
                "legacy 0.050877, core 0.0402124",
            ),
            (
                "normal_boolean|imbalanced_boolean",
                "chi2",
                "legacy 0.5142, core 0.609817",
            ),
            (
                "normal_boolean|imbalanced_boolean",
                "cramers_v",
                "legacy 0.0099, core 0.0107775",
            ),
            (
                "normal_boolean|imbalanced_boolean",
                "p_value",
                "legacy 0.473337, core 0.434857",
            ),
            (
                "normal_boolean|imbalanced_target",
                "chi2",
                "legacy 0.4922, core 0.588918",
            ),
            (
                "normal_boolean|imbalanced_target",
                "cramers_v",
                "legacy 0.0097, core 0.0105913",
            ),
            (
                "normal_boolean|imbalanced_target",
                "p_value",
                "legacy 0.482934, core 0.442838",
            ),
            ("normal_boolean|mostly_zeros", "chi2", "legacy 0.1197, core 0.153893"),
            (
                "normal_boolean|mostly_zeros",
                "cramers_v",
                "legacy 0.0048, core 0.00541414",
            ),
            (
                "normal_boolean|mostly_zeros",
                "p_value",
                "legacy 0.729395, core 0.694843",
            ),
            ("normal_boolean|perfect_leakage", "chi2", "legacy 1.2247, core 1.29185"),
            (
                "normal_boolean|perfect_leakage",
                "cramers_v",
                "legacy 0.0153, core 0.0156865",
            ),
            (
                "normal_boolean|perfect_leakage",
                "p_value",
                "legacy 0.268441, core 0.255707",
            ),
            (
                "normal_boolean|target_for_leakage",
                "chi2",
                "legacy 1.2247, core 1.29185",
            ),
            (
                "normal_boolean|target_for_leakage",
                "cramers_v",
                "legacy 0.0153, core 0.0156865",
            ),
            (
                "normal_boolean|target_for_leakage",
                "p_value",
                "legacy 0.268441, core 0.255707",
            ),
            ("target_for_leakage|perfect_leakage", "chi2", "legacy 5245.3, core 5250"),
            (
                "target_for_leakage|perfect_leakage",
                "cramers_v",
                "legacy 0.9996, core 1",
            ),
        ],
    ),
    *_group(
        "contingency_tables",
        "titanic",
        YATES,
        [
            ("adult_male|alive", "chi2", "legacy 274.175, core 276.511"),
            ("adult_male|alive", "cramers_v", "legacy 0.5547, core 0.55708"),
            ("adult_male|alone", "chi2", "legacy 144.276, core 145.961"),
            ("adult_male|alone", "cramers_v", "legacy 0.4024, core 0.404744"),
            ("alive|alone", "chi2", "legacy 36.0005, core 36.8501"),
            ("alive|alone", "cramers_v", "legacy 0.201, core 0.203367"),
            ("sex|adult_male", "chi2", "legacy 731.652, core 735.534"),
            ("sex|adult_male", "cramers_v", "legacy 0.9062, core 0.908578"),
            ("sex|alive", "chi2", "legacy 260.717, core 263.051"),
            ("sex|alive", "cramers_v", "legacy 0.5409, core 0.543351"),
            ("sex|alone", "chi2", "legacy 80.8573, core 82.1511"),
            ("sex|alone", "cramers_v", "legacy 0.3012, core 0.303646"),
            ("survived|adult_male", "chi2", "legacy 274.175, core 276.511"),
            ("survived|adult_male", "cramers_v", "legacy 0.5547, core 0.55708"),
            ("survived|alive", "chi2", "legacy 886.777, core 891"),
            ("survived|alive", "cramers_v", "legacy 0.9976, core 1"),
            ("survived|alone", "chi2", "legacy 36.0005, core 36.8501"),
            ("survived|alone", "cramers_v", "legacy 0.201, core 0.203367"),
            ("survived|sex", "chi2", "legacy 260.717, core 263.051"),
            ("survived|sex", "cramers_v", "legacy 0.5409, core 0.543351"),
        ],
    ),
    *_group(
        "contingency_tables",
        "wide",
        CORRECTED,
        [
            ("bool_04", "finding", "legacy only (info)"),
            ("cat_bal_20", "finding", "legacy only (info)"),
        ],
    ),
    *_group(
        "contingency_tables",
        "wide",
        YATES,
        [
            ("bool_00|bool_01", "chi2", "legacy 0.4342, core 0.525907"),
            ("bool_00|bool_01", "cramers_v", "legacy 0.0208, core 0.0229327"),
            ("bool_00|bool_01", "p_value", "legacy 0.509925, core 0.468333"),
            ("bool_00|bool_02", "chi2", "legacy 0.0008, core 0.00952468"),
            ("bool_00|bool_02", "cramers_v", "legacy 0.0009, core 0.00308621"),
            ("bool_00|bool_02", "p_value", "legacy 0.977439, core 0.922254"),
            ("bool_00|bool_03", "chi2", "legacy 0.6135, core 0.721178"),
            ("bool_00|bool_03", "cramers_v", "legacy 0.0248, core 0.0268548"),
            ("bool_00|bool_03", "p_value", "legacy 0.433458, core 0.395758"),
            ("bool_00|bool_04", "chi2", "legacy 0.3213, core 0.406459"),
            ("bool_00|bool_04", "cramers_v", "legacy 0.0179, core 0.0201608"),
            ("bool_00|bool_04", "p_value", "legacy 0.570809, core 0.523772"),
            ("bool_01|bool_02", "chi2", "legacy 0.3002, core 0.37998"),
            ("bool_01|bool_02", "cramers_v", "legacy 0.0173, core 0.0194931"),
            ("bool_01|bool_02", "p_value", "legacy 0.58378, core 0.537614"),
            ("bool_01|bool_03", "chi2", "legacy 0.0053, core 0.0191111"),
            ("bool_01|bool_03", "cramers_v", "legacy 0.0023, core 0.00437162"),
            ("bool_01|bool_03", "p_value", "legacy 0.941779, core 0.890048"),
            ("bool_01|bool_04", "chi2", "legacy 0.2213, core 0.291988"),
            ("bool_01|bool_04", "cramers_v", "legacy 0.0149, core 0.0170877"),
            ("bool_01|bool_04", "p_value", "legacy 0.63803, core 0.588949"),
            ("bool_02|bool_03", "chi2", "legacy 1.4227, core 1.5901"),
            ("bool_02|bool_03", "cramers_v", "legacy 0.0377, core 0.039876"),
            ("bool_02|bool_03", "p_value", "legacy 0.232965, core 0.207312"),
            ("bool_02|bool_04", "cramers_v", "legacy 0, core 0.000185053"),
            ("bool_02|bool_04", "p_value", "legacy 1, core 0.995331"),
            ("bool_03|bool_04", "chi2", "legacy 0.251, core 0.325541"),
            ("bool_03|bool_04", "cramers_v", "legacy 0.0158, core 0.0180428"),
            ("bool_03|bool_04", "p_value", "legacy 0.616384, core 0.568297"),
        ],
    ),
    *_group(
        "detect_data_leakage",
        "all_categorical",
        EMPTY_RESULT,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "detect_data_leakage",
        "kitchen5k",
        CODED,
        [
            ("perfect_leakage|subtle_leakage", "pair", "legacy only"),
            ("target_for_leakage|perfect_leakage", "pair", "legacy only"),
            ("target_for_leakage|subtle_leakage", "pair", "legacy only"),
        ],
    ),
    *_group(
        "detect_data_leakage",
        "single_column",
        EMPTY_RESULT,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "compute_pairwise_associations",
        "sms_spam",
        ONE_CATEGORICAL,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "contingency_tables",
        "sms_spam",
        ONE_CATEGORICAL,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
    *_group(
        "detect_data_leakage",
        "sms_spam",
        ONE_CATEGORICAL,
        [
            ("", "status", "legacy ok, core not_applicable"),
        ],
    ),
)
