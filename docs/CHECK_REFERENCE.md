# DSBF check reference

> **Generated file.** Do not edit it. `dsbf docs export` writes it from the core
> pack's check specs, parameter models and sources, and `dsbf docs export --check`
> fails when it is stale.

Each section lists what a check applies to, what it needs, its parameters and the
finding codes it can raise. A parameter is set by check id with
`dsbf.profile(params={"<check id>": {"<parameter>": value}})` or
`dsbf profile --param <check id>.<parameter>=value`. The check's own text
carries the limits of its method.

## Checks

| Check | Kind | Cost | Summary |
| --- | --- | --- | --- |
| [`associations`](#associations) | analysis | moderate | Measure each numeric column against every other numeric and categorical one. |
| [`autocorrelation`](#autocorrelation) | analysis | moderate | Compute the ACF and PACF of each numeric column on a regular series. |
| [`bimodality`](#bimodality) | analysis | moderate | Flag numeric columns that two Gaussian peaks fit clearly better than one. |
| [`categorical_encoding`](#categorical_encoding) | rule | cheap | Suggest an encoding for each categorical column from its cardinality. |
| [`class_imbalance`](#class_imbalance) | rule | cheap | Flag a target whose most frequent class is at least `majority_share` of it. |
| [`collinearity`](#collinearity) | analysis | moderate | Measure how well each numeric column is predicted by all the others. |
| [`constant_columns`](#constant_columns) | rule | cheap | Flag columns in which every non-null value is the same. |
| [`contingency`](#contingency) | analysis | moderate | Test each pair of categorical columns for independence and size the link. |
| [`datetime_consistency`](#datetime_consistency) | rule | cheap | Report the values of a datetime column that its inferred format cannot read. |
| [`datetime_features`](#datetime_features) | rule | cheap | List the calendar features that vary in each datetime column. |
| [`dominant_value`](#dominant_value) | rule | cheap | Flag columns in which the most frequent value is most of the non-null rows. |
| [`dtype_optimizations`](#dtype_optimizations) | rule | cheap | Suggest a narrower type for integer, two-valued and low-cardinality columns. |
| [`duplicate_columns`](#duplicate_columns) | rule | cheap | Group the columns that have the same values in the same rows. |
| [`duplicate_rows`](#duplicate_rows) | rule | cheap | Report the rows that repeat an earlier row, in every column. |
| [`encoded_columns`](#encoded_columns) | analysis | cheap | Report columns whose values are UUIDs, digests, base64 or random tokens. |
| [`entropy`](#entropy) | rule | cheap | Report the Shannon entropy of each categorical, boolean and text column. |
| [`format_violations`](#format_violations) | analysis | cheap | Count the values that break each rule, and find dates written two ways. |
| [`fuzzy_duplicates`](#fuzzy_duplicates) | analysis | heavy | Find pairs of rows whose text is almost, but not exactly, the same. |
| [`group_differences`](#group_differences) | analysis | moderate | Test whether each numeric column differs between the groups of each category. |
| [`high_cardinality`](#high_cardinality) | rule | cheap | Flag categorical columns with more distinct values than `max_distinct`. |
| [`id_columns`](#id_columns) | rule | cheap | Flag columns whose values label rows rather than describe them. |
| [`imputation`](#imputation) | synthesis | cheap | Suggest how to fill each column's nulls from the null rate and its mechanism. |
| [`kurtosis`](#kurtosis) | rule | cheap | Flag numeric columns with heavy or thin tails, from their excess kurtosis. |
| [`missingness_mechanism`](#missingness_mechanism) | analysis | moderate | Test what the rows with a null in a column have in common. |
| [`mixed_types`](#mixed_types) | analysis | cheap | Report columns that mix numbers, dates, booleans and words. |
| [`ml_readiness`](#ml_readiness) | synthesis | cheap | Report the work a dataset needs before modelling. |
| [`multivariate_outliers`](#multivariate_outliers) | analysis | moderate | Rank the rows by how easily an isolation forest separates them from the rest. |
| [`mutual_information`](#mutual_information) | analysis | moderate | Rank the features by what they tell about the target, linear or not. |
| [`near_zero_variance`](#near_zero_variance) | rule | cheap | Flag numeric columns whose variance is at or below `max_variance`. |
| [`normality`](#normality) | analysis | moderate | Test numeric columns against a normal distribution and keep their Q-Q data. |
| [`null_rate`](#null_rate) | rule | cheap | Report the share of missing values in each column. |
| [`numeric_binning`](#numeric_binning) | rule | cheap | Suggest quantile or equal-width bins, and how many, for each numeric column. |
| [`out_of_bounds`](#out_of_bounds) | analysis | cheap | Count the values of each bounded numeric column outside its range. |
| [`outliers`](#outliers) | analysis | moderate | Flag numeric values beyond the IQR fences, a z-score or a modified z-score. |
| [`quality_dimensions`](#quality_dimensions) | synthesis | cheap | Report completeness, validity, usability, redundancy and leakage. |
| [`reference_drift`](#reference_drift) | analysis | moderate | Compare each column's type, nulls and distribution with the reference's. |
| [`sample_size`](#sample_size) | rule | cheap | Compare the rows with what rules of thumb ask for the number of features. |
| [`schema`](#schema) | analysis | cheap | Compare the columns with the expected types, nulls, ranges and values. |
| [`seasonal_decomposition`](#seasonal_decomposition) | analysis | moderate | Split each numeric column into trend, seasonal and remainder by STL. |
| [`skewness`](#skewness) | rule | cheap | Flag numeric columns that lean to one side or whose mean sits off the median. |
| [`stationarity`](#stationarity) | analysis | moderate | Test each numeric column for a unit root and a drifting level over time. |
| [`string_anomalies`](#string_anomalies) | analysis | cheap | Find values that differ only by case, padding, hidden characters or length. |
| [`target_leakage`](#target_leakage) | analysis | moderate | Flag features that predict the target almost perfectly, or follow it in time. |
| [`temporal_gaps`](#temporal_gaps) | rule | cheap | Flag missing periods and irregular spacing in datetime columns. |
| [`text_encoding`](#text_encoding) | rule | cheap | Report names and values that hold U+FFFD or NUL characters. |
| [`transformations`](#transformations) | analysis | moderate | Try log, root, square, Box–Cox and Yeo–Johnson on each skewed column. |
| [`zeros`](#zeros) | rule | cheap | Flag numeric columns in which zero is a large share of the values. |

## associations

Measure each numeric column against every other numeric and categorical one.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | numeric or categorical or boolean |
| Dimensions | relationship, redundancy |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `method` | `'auto'` | The statistic for a numeric pair. `auto` uses Kendall's τ-b below `small_sample` complete rows and Pearson's r from there on. |
| `small_sample` | `30` | Complete rows below which `auto` uses Kendall's τ. (≥ 5) |
| `min_n` | `8` | Complete rows a pair needs. Routing to a statistic happens after this gate, never before it, so a small frame gets Kendall's τ. (≥ 4) |
| `max_levels` | `50` | Most levels a categorical column may have; η over more levels is inflated by the groups' own noise. (≥ 2) |
| `interval_level` | `0.95` | Coverage of the Fisher z intervals. (> 0) (< 1) |
| `redundancy_threshold` | `0.99` | \|Pearson r\| from which two numeric columns are redundant. (> 0) (≤ 1) |
| `max_pairs` | `10000` | Most pairs tested; with more, the check is skipped rather than test an arbitrary part of them. (≥ 1) |
| `max_rows` | `200000` | Most rows read; a larger frame is sampled with `seed`. (≥ 1000) |
| `seed` | `42` | Seed of the row sample. |
| `max_findings` | `50` | Most findings; the redundant and strongest pairs first. (≥ 1) |

**Output**

`AssociationsOutput`: `pairs`, `pairs_tested`, `pairs_flagged`, `untested`, `skipped_columns`, `rows_used`, `rows_total`

**Finding codes**

`redundant_columns`, `strong_association`

## autocorrelation

Compute the ACF and PACF of each numeric column on a regular series.

The columns are resampled to the frequency first (see ``checks.timeseries``), the
most recent ``max_periods`` periods are used, and the lags are capped.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | numeric or datetime |
| Dimensions | temporal |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | time_index |
| Needs reference | no |
| Extras | timeseries |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `time_index` | `None` | The datetime column that orders the rows. `dsbf.profile(time_index=...)` sets it; without one the check is not applicable. |
| `frequency` | `None` | The spacing of the series, as an offset alias: a count and a unit (`D`, `h`, `min`, `s`, `ms`, `W`: fixed spans; `MS`, `ME`, `QS`, `QE`, `YS`, `YE`: calendar months), such as `15min` or `2W`. Unset, the spacing the profile found in the index is used. |
| `aggregate` | `'mean'` | How the values of one period are combined, which is what repeated timestamps and rows finer than the frequency need. |
| `max_periods` | `2000` | Most periods used: the most recent ones. Tests on tens of thousands of points reject trivial deviations, and the cost of the lag regressions grows with them. (≥ 30) (≤ 100000) |
| `max_fill_share` | `0.2` | Largest share of a series' periods that may be empty and interpolated; a column with more has no series. (≥ 0) (≤ 0.9) |
| `value_columns` | `()` | The numeric columns to analyse; empty is every numeric column. |
| `alpha` | `0.05` | Significance level of the confidence bands: Bartlett's for the ACF, ±z/√n for the PACF. (> 0) (< 1) |
| `omnibus_alpha` | `0.01` | Level of the Ljung-Box test over the first lags that a column must fail before any one lag is reported. With 20 or 40 lags a lag lies outside its band by chance in most white-noise series, so a lag counts only in a series that is not white noise as a whole. (> 0) (< 1) |
| `max_lags` | `40` | Most lags computed; n periods allow at most n // 2 - 1. (≥ 2) (≤ 365) |
| `min_abs` | `0.2` | Smallest \|autocorrelation\| a lag is flagged at, besides lying outside the band: over many periods a negligible lag is significant. (≥ 0) (< 1) |
| `min_periods` | `20` | Periods a series needs to be analysed. (≥ 10) |

**Output**

`AutocorrelationOutput`: `index`, `frequency`, `frequency_source`, `aggregate`, `max_periods`, `periods_total`, `truncated`, `timezone`, `skipped`, `alpha`, `min_abs`, `columns`

**Finding codes**

`autocorrelated`

## bimodality

Flag numeric columns that two Gaussian peaks fit clearly better than one.

Peaks 4 spreads apart are found in over 80% of columns of 100 values and in all
of 200 or more; 3 spreads apart, only in about half of 1 000. Under about 100
values the BIC cannot tell two peaks from one, so a column that short is rarely
flagged, and a uniform column of that size is flagged by chance in about 7% of
samples (under 3% from 200 values).

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | numeric |
| Dimensions | distribution |
| Needs profile | counts, numeric_stats |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | ml |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `bic_threshold` | `10.0` | Improvement in BIC from a second component above which it is worth having. Raftery's scale calls 10 'very strong'. (≥ 0) |
| `min_separation` | `2.0` | Ashman's D between the components, from which they are two peaks and not one skewed or heavy-tailed one. (> 0) |
| `max_valley_ratio` | `0.75` | Most the histogram may dip between the peaks, as the density at the valley over that at the lower peak. A flat or one-humped column has a ratio near 1, and a mixture fits it as two Gaussians all the same. (> 0) (≤ 1) |
| `min_weight` | `0.05` | Share of the values the smaller component must hold. (> 0) (< 0.5) |
| `strong_bic_per_value` | `0.05` | Improvement in BIC per value from which, with a clean separation, the finding is a warning. (> 0) |
| `skewness_guard` | `2.0` | \|skewness\| above which a column is not fitted: a long tail looks like a second component. (> 0) |
| `min_n` | `10` | Finite values a column needs. Below about 100 the BIC rarely separates two peaks, even 4 spreads apart. (≥ 4) |
| `max_rows` | `10000` | Most values fitted; a larger column is sampled with `seed`. (≥ 1000) |
| `seed` | `42` | Seed of the mixture fits and of the sample. |

**Output**

`BimodalityOutput`: `columns`, `skipped`

**Finding codes**

`bimodal`

## categorical_encoding

Suggest an encoding for each categorical column from its cardinality.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | categorical |
| Dimensions | usability |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `max_one_hot` | `10` | Most distinct values for which one-hot columns are suggested. (≥ 2) |
| `min_rows_per_level` | `20` | Rows per distinct value below which one-hot columns would be too sparse to estimate, so the frequency of each value is suggested. (≥ 1) |
| `target_folds` | `5` | Folds of the out-of-fold target encoding suggested for a column that predicts the target. (≥ 2) |
| `target_smoothing` | `20` | Rows of prior the target encoding adds to each value, which pulls the mean of a rare value toward the overall mean. (> 0) |

**Output**

`CategoricalEncodingOutput`: `columns`

**Finding codes**

`encoding_strategy`

## class_imbalance

Flag a target whose most frequent class is at least `majority_share` of it.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | categorical or boolean |
| Dimensions | distribution |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | target |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `target` | `None` | The column to predict. `dsbf.profile(target=...)` sets it; without one the check is not applicable. |
| `majority_share` | `0.9` | Share of the non-null target from which the majority class makes the target imbalanced. (> 0) (≤ 1) |

**Output**

`ClassImbalanceOutput`: `target`, `n_classes`, `majority_class`, `majority_count`, `majority_share`, `minority_class`, `minority_count`, `imbalance_ratio`, `imbalanced`

**Finding codes**

`class_imbalance`

## collinearity

Measure how well each numeric column is predicted by all the others.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | numeric |
| Dimensions | redundancy |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `vif_threshold` | `10.0` | VIF from which a column is a warning. (> 1) |
| `moderate_threshold` | `5.0` | VIF from which a column is reported at all. (> 1) |
| `max_missing_share` | `0.4` | Share of missing values above which a column is left out before any row is dropped, so a sparse column does not cost every row. (≥ 0) (< 1) |
| `min_rows` | `10` | Complete rows needed, at least the number of columns plus one. (≥ 3) |
| `max_columns` | `200` | Most columns; with more, the check is skipped rather than read an arbitrary part of them. (≥ 2) |
| `max_rows` | `100000` | Most rows read; a larger frame is sampled with `seed`. (≥ 1000) |
| `seed` | `42` | Seed of the row sample. |
| `max_findings` | `50` | Most findings; the highest VIFs first. (≥ 1) |

**Output**

`CollinearityOutput`: `definition`, `columns`, `n_used`, `n_columns`, `rows_total`, `rows_read`, `columns_flagged`, `skipped_columns`, `dropped_for_rows`, `condition_number`

**Finding codes**

`multicollinearity`

## constant_columns

Flag columns in which every non-null value is the same.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | any type |
| Dimensions | usability |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

None.

**Output**

`ConstantColumnsOutput`: `constant_columns`

**Finding codes**

`constant_column`

## contingency

Test each pair of categorical columns for independence and size the link.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | categorical or boolean |
| Dimensions | relationship |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `alpha` | `0.05` | Significance level of the chi-squared test. (> 0) (< 1) |
| `min_v` | `0.1` | Bias-corrected V from which a significant pair is reported and gets a bootstrap interval. (> 0) (< 1) |
| `min_n` | `10` | Complete rows a pair needs. (≥ 5) |
| `max_levels` | `50` | Most levels a column may have to be compared. (≥ 2) |
| `top_n` | `10` | Levels per column printed in a pair's table; the statistics always use all the levels. (≥ 2) |
| `interval_level` | `0.95` | Coverage of the bootstrap interval. (> 0) (< 1) |
| `n_boot` | `500` | Bootstrap resamples of a table. (≥ 100) |
| `seed` | `42` | Seed of the bootstrap. |
| `max_pairs` | `5000` | Most pairs tested; with more, the check is skipped rather than test an arbitrary part of them. (≥ 1) |
| `max_findings` | `50` | Most findings; the strongest pairs first. (≥ 1) |

**Output**

`ContingencyOutput`: `pairs`, `pairs_tested`, `pairs_flagged`, `untested`, `skipped_columns`

**Finding codes**

`categorical_association`

## datetime_consistency

Report the values of a datetime column that its inferred format cannot read.

The profile typed a text column datetime only if one format reads over 90% of its
values, and counted the others as unparsed, so a share of 10% or more cannot occur
here; a column that is mostly unreadable is not a datetime column. A date or
datetime column has none.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | datetime |
| Dimensions | validity |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `warn_share` | `0.05` | Share of a column's non-null values that do not read, from which the finding is a warning rather than information. (> 0) (≤ 1) |

**Output**

`DatetimeConsistencyOutput`: `columns`

**Finding codes**

`unparsed_values`

## datetime_features

List the calendar features that vary in each datetime column.

A model cannot use a raw timestamp; its signal is in parts of it. A part is
listed only if the column varies in it: no hour features for midnight dates, no
month features inside one month.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | datetime |
| Dimensions | usability |
| Needs profile | counts, datetime |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

None.

**Output**

`DatetimeFeaturesOutput`: `columns`

**Finding codes**

`extractable_features`

## dominant_value

Flag columns in which the most frequent value is most of the non-null rows.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | categorical or boolean |
| Dimensions | distribution |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `notice_share` | `0.7` | Share from which a value is reported. (> 0) (≤ 1) |
| `warn_share` | `0.9` | Share from which the finding is a warning. (> 0) (≤ 1) |
| `flag_share` | `0.95` | Share from which a column is flagged. (> 0) (≤ 1) |

**Output**

`DominantValueOutput`: `columns`

**Finding codes**

`dominant_value`

## dtype_optimizations

Suggest a narrower type for integer, two-valued and low-cardinality columns.

The profile holds no memory per column, so a saving is estimated from the type's
width times the rows, and for strings from the mean length plus a 4-byte offset
per value, against a dictionary of the distinct values and one small code per
row. A column is narrowed only when the profile shows its values fit: an integer
column's range is its minimum and maximum, or, for a column of codes the profile
types categorical, its values when it lists them all. Floating point columns are
never suggested: nothing in the profile shows float32 holds their precision.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | numeric or categorical |
| Dimensions | usability |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `min_saving_bytes` | `1024` | The estimated saving below which a column is not reported, so a small column does not add noise. (≥ 0) |

**Output**

`DtypeOptimizationsOutput`: `suggestions`, `total_saving_bytes`

**Finding codes**

`smaller_dtype`

## duplicate_columns

Group the columns that have the same values in the same rows.

Two columns are equal when their 64-bit content digests and storage types are. The
digest follows row order, so the same values in another order do not match. A
chance match between two columns is about one in 2^64.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | any type |
| Dimensions | redundancy |
| Needs profile | counts, content_hash |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

None.

**Output**

`DuplicateColumnsOutput`: `groups`, `pairs`

**Finding codes**

`duplicate_columns`

## duplicate_rows

Report the rows that repeat an earlier row, in every column.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | the dataset |
| Dimensions | validity |
| Needs profile | duplicate_rows |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `warn_share` | `0.01` | Share of rows above which the finding is a warning. (≥ 0) (≤ 1) |

**Output**

`DuplicateRowsOutput`: `duplicate_rows`, `rows`, `duplicate_share`

**Finding codes**

`duplicate_rows`

## encoded_columns

Report columns whose values are UUIDs, digests, base64 or random tokens.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | analysis |
| Cost | cheap |
| Applies to | categorical or text or identifier |
| Dimensions | usability |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `min_values` | `10` | Non-null values a column needs to be read. (≥ 1) |
| `min_share` | `0.9` | Share of a column's values that must look encoded. (> 0) (≤ 1) |
| `detect_uuid` | `True` | Look for UUIDs. |
| `detect_hex` | `True` | Look for hexadecimal digests of 32 to 128 digits, and shorter hexadecimal keys (see `min_hex_length`). |
| `min_hex_length` | `8` | Digits a hexadecimal key of another length than a digest has at least, when it holds both a digit and a letter: a request id or a short commit hash. Random hexadecimal has no digit or no letter in about one value in fifty at 8 digits and almost never at more. (≥ 4) |
| `detect_base64` | `True` | Look for base64 strings. |
| `detect_tokens` | `True` | Look for opaque tokens: long strings of one length with a high entropy per character. |
| `min_base64_length` | `12` | Characters a base64 string has at least; shorter words that mix cases and digits are too often plain text. (≥ 4) |
| `min_token_length` | `16` | Characters an opaque token has at least. (≥ 4) |
| `min_entropy` | `5.0` | Shannon entropy, in bits per character, of the characters of all the candidate tokens together from which they look random. Random letters and digits have 5.2 to 6, email addresses and other text 4.1 to 4.7. (> 0) |
| `length_std_threshold` | `2.0` | Largest standard deviation of the length of the opaque tokens: hashes and keys are of one length. (≥ 0) |
| `max_examples` | `3` | Examples listed per column. (≥ 1) (≤ 10) |

**Output**

`EncodedColumnsOutput`: `columns`, `columns_read`, `skipped_columns`

**Finding codes**

`encoded_values`

## entropy

Report the Shannon entropy of each categorical, boolean and text column.

This raises no finding: a skewed column is `dominant_value`'s to report, and a
column of one value is `constant_columns`'s. The values feed the distribution view.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | categorical or text or boolean |
| Dimensions | distribution |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `min_rows` | `30` | Non-null values below which the estimates are called unstable: rare values may be missing, so the entropy is underestimated. (≥ 1) |

**Output**

`EntropyOutput`: `columns`

**Finding codes**

None.

## format_violations

Count the values that break each rule, and find dates written two ways.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | analysis |
| Cost | cheap |
| Applies to | any type |
| Dimensions | validity |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `rules` | `()` | The rules to check, one per column. With none, only the mixed date formats are reported. |
| `check_date_formats` | `True` | Report string columns that write dates in several formats. |
| `min_date_share` | `0.9` | Share of a column's values that must be dates for it to be read as a column of dates. (> 0) (≤ 1) |
| `min_format_share` | `0.05` | Share of the values a date format needs to count as in use. (> 0) (≤ 0.5) |
| `min_values` | `10` | Values a column needs for the date formats. (≥ 1) |
| `warn_share` | `0.01` | Share of a column's values that break its rule above which the finding is a warning. (≥ 0) (≤ 1) |
| `max_examples` | `5` | Examples listed per rule. (≥ 1) (≤ 20) |

**Output**

`FormatViolationsOutput`: `rules`, `unchecked_rules`, `mixed_dates`, `columns_read`

**Finding codes**

`format_violations`, `mixed_date_formats`

## fuzzy_duplicates

Find pairs of rows whose text is almost, but not exactly, the same.

| | |
| --- | --- |
| Version | 1.2.1 |
| Kind | analysis |
| Cost | heavy |
| Applies to | categorical or text or identifier |
| Dimensions | validity |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | text |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `threshold` | `0.92` | Similarity from which two records are near-duplicates: the mean of the normalised Indel ratio of the two values, over the columns either has a value in, weighted by each column's entropy in bits, so one long shared field or a column that hardly varies cannot make unlike rows alike. (≥ 0.5) (≤ 1.0) |
| `number_cap` | `0.7` | The most two values alike when both hold digits and the digits differ. Another number (an email's serial, a street number, a sequential code) is evidence of a different record, which a typo in letters is not; 1 reads digits as letters, so a typo in a number is a typo. (≥ 0) (≤ 1) |
| `max_label_levels` | `20` | A column with at most this many distinct values is a label and is left out of the row string: it says what kind of record a row is, not which one. (≥ 1) |
| `max_chars` | `200` | Characters of a value that are compared; the rest is cut. (≥ 20) (≤ 5000) |
| `max_rows` | `50000` | Most distinct records compared; more are sampled with `seed`. (≥ 10) |
| `max_pairs` | `2000000` | Most pairs scored. When every pair fits all are scored, otherwise only the pairs that share a rare token. (≥ 100) |
| `max_block_size` | `50` | Most records that may share a token for it to block: a token shared by more says nothing about which records belong together. (≥ 2) (≤ 1000) |
| `max_seconds` | `60.0` | Seconds of scoring after which the check stops. (> 0) |
| `max_pairs_listed` | `20` | Most similar pairs listed in the output. (≥ 1) (≤ 200) |
| `warn_share` | `0.01` | Share of rows in a flagged pair above which the finding is a warning. (≥ 0) (≤ 1) |
| `exclude_columns` | `()` | Columns left out of the comparison, such as a key that differs between copies of a record. |
| `seed` | `42` | Seed of the sample taken above `max_rows`. |

**Output**

`FuzzyDuplicatesOutput`: `columns`, `column_weights`, `skipped_columns`, `rows`, `records`, `records_compared`, `exact_repeats`, `blocking`, `pairs_possible`, `pairs_compared`, `coverage`, `stopped_by`, `threshold`, `pairs_flagged`, `rows_involved`, `top_pairs`

**Finding codes**

`near_duplicate_rows`

## group_differences

Test whether each numeric column differs between the groups of each category.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | numeric or categorical or boolean |
| Dimensions | relationship |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `alpha` | `0.05` | Level of the adjusted p-value for a difference. (> 0) (< 1) |
| `correction` | `'fdr_bh'` | Adjustment of the p-values for the number of pairs tested, separately for each method. `fdr_bh` controls the false discovery rate. |
| `equal_variances` | `False` | Take the p-values from the classic ANOVA F and the Kruskal–Wallis H, which assume the groups have the same spread. Off, they come from Welch's ANOVA on the values and on their ranks, which do not: with groups of unequal size and spread, the classic tests call equal groups different half the time (5% is the promise). |
| `min_effect` | `0.01` | Effect size (ω² or ε²) from which a significant pair is reported; a huge sample makes a negligible difference significant. (≥ 0) (< 1) |
| `min_group_n` | `5` | Rows a group needs; a smaller group is left out. (≥ 2) |
| `max_levels` | `20` | Most levels a categorical column may have. (≥ 2) |
| `max_contrast_levels` | `10` | Most groups for which a flagged pair gets every pairwise Mann–Whitney contrast. (≥ 2) |
| `interval_level` | `0.95` | Coverage of the bootstrap intervals. (> 0) (< 1) |
| `n_boot` | `200` | Bootstrap resamples of a pair. (≥ 100) |
| `seed` | `42` | Seed of the bootstrap and of the row sample. |
| `max_pairs` | `5000` | Most pairs tested; with more, the check is skipped rather than test an arbitrary part of them. (≥ 1) |
| `max_rows` | `200000` | Most rows read; a larger frame is sampled with `seed`. (≥ 1000) |
| `max_findings` | `50` | Most findings; the largest effects first. (≥ 1) |

**Output**

`GroupDifferencesOutput`: `pairs`, `pairs_tested`, `pairs_flagged`, `tests_run`, `correction`, `alpha`, `untested`, `skipped_columns`, `rows_used`, `rows_total`

**Finding codes**

`group_difference`

## high_cardinality

Flag categorical columns with more distinct values than `max_distinct`.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | categorical |
| Dimensions | usability |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `max_distinct` | `50` | Distinct values above which a categorical column is reported. (≥ 2) |

**Output**

`HighCardinalityOutput`: `columns`

**Finding codes**

`high_cardinality`

## id_columns

Flag columns whose values label rows rather than describe them.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | identifier or categorical or text or numeric |
| Dimensions | usability |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `min_unique_share` | `0.95` | Distinct values per non-null value from which a string or categorical column is read as an identifier. (> 0) (≤ 1) |
| `min_rows` | `20` | Non-null values a column needs before its uniqueness means anything. (≥ 2) |

**Output**

`IdColumnsOutput`: `id_columns`

**Finding codes**

`likely_identifier`

## imputation

Suggest how to fill each column's nulls from the null rate and its mechanism.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | synthesis |
| Cost | cheap |
| Applies to | the dataset |
| Dimensions | usability |
| Needs profile | nothing |
| Needs checks | null_rate |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `min_null_share` | `0.01` | Share of rows that must be null for a column to be advised. (≥ 0) (< 1) |
| `skew_threshold` | `1.0` | Absolute skewness from which the median, not the mean, fills a numeric column. (> 0) |

**Output**

`ImputationOutput`: `columns`

**Finding codes**

`imputation_strategy`

## kurtosis

Flag numeric columns with heavy or thin tails, from their excess kurtosis.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | numeric |
| Dimensions | distribution |
| Needs profile | counts, numeric_stats |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `mild` | `1.0` | Excess kurtosis from which tails are called heavier than normal. |
| `strong` | `3.0` | Excess kurtosis from which tails are heavy and the finding is a warning. |
| `thin` | `-1.0` | Excess kurtosis at or below which tails are called thin. |

**Output**

`KurtosisOutput`: `columns`

**Finding codes**

`heavy_tails`, `thin_tails`

## missingness_mechanism

Test what the rows with a null in a column have in common.

The null of each column is a 0/1 indicator, compared with every other column on
the rows where that column has a value: numbers by Mann–Whitney, levels by
chi-squared. Every p-value is adjusted together. A dependence says the nulls are
not a random sample of the rows (not MCAR); it cannot tell missing at random
from missing not at random, which depends on the unobserved values.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | any type |
| Dimensions | relationship |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `alpha` | `0.05` | Level of the adjusted p-value for a dependence. (> 0) (< 1) |
| `correction` | `'fdr_bh'` | Adjustment of the p-values for the number of tests run over all the columns with nulls and all their predictors. `fdr_bh` controls the false discovery rate. |
| `min_effect` | `0.1` | Effect size (\|rank-biserial r\| or Cramér's V) from which a significant test is reported; a huge sample makes a weak dependence significant. (≥ 0) (< 1) |
| `min_null_share` | `0.01` | Share of rows that must be null for a column to be analysed. (≥ 0) (< 1) |
| `min_group_n` | `8` | Rows each of the missing and observed groups needs for a test. (≥ 2) |
| `max_levels` | `20` | Most levels a categorical predictor may have. (≥ 2) |
| `max_subjects` | `30` | Most columns with nulls analysed, those with most nulls first. (≥ 1) |
| `max_rows` | `200000` | Most rows read; a larger frame is sampled with `seed`. (≥ 1000) |
| `seed` | `42` | Seed of the row sample. |
| `max_listed` | `10` | Most dependences listed for one column. (≥ 1) |
| `min_report_share` | `0.05` | Null share from which a column with no dependence found is still reported as a finding. (≥ 0) (< 1) |
| `warn_null_share` | `0.2` | Null share above which a strong dependence is a warning. (> 0) (< 1) |

**Output**

`MissingnessMechanismOutput`: `columns`, `tests_run`, `tests_flagged`, `correction`, `rows_used`, `skipped_columns`, `unusable_predictors`

**Finding codes**

`missingness_depends_on_observed`, `missingness_unexplained`

## mixed_types

Report columns that mix numbers, dates, booleans and words.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | analysis |
| Cost | cheap |
| Applies to | categorical or text or identifier |
| Dimensions | validity |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `min_share` | `0.05` | Share of the values a kind needs to count as present. (> 0) (≤ 0.5) |
| `ignore_null_tokens` | `True` | Leave out values that stand for a missing value (`NA`, `null`, `none`, `-`, an empty or blank string). Off, they are a kind of their own. |
| `min_values` | `10` | Values a column needs to be read. (≥ 1) |
| `max_examples` | `3` | Examples listed per kind. (≥ 1) (≤ 10) |

**Output**

`MixedTypesOutput`: `columns`, `columns_read`, `skipped_columns`

**Finding codes**

`mixed_value_types`

## ml_readiness

Report the work a dataset needs before modelling.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | synthesis |
| Cost | cheap |
| Applies to | the dataset |
| Dimensions | usability |
| Needs profile | nothing |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

None.

**Output**

`MlReadinessOutput`: `gate`, `complete`, `unassessed`, `total_columns`, `dimensions`, `action_columns`, `advisory_columns`, `clean_columns`

**Finding codes**

None.

## multivariate_outliers

Rank the rows by how easily an isolation forest separates them from the rest.

| | |
| --- | --- |
| Version | 1.2.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | numeric |
| Dimensions | distribution |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | ml |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `contamination` | `0.05` | Share of the rows the forest ranks as most anomalous. It flags exactly this share of whatever it is given, so the count is only a ranking; whether any row stands out is `alpha` and `min_excess`. (> 0) (≤ 0.5) |
| `alpha` | `0.01` | Chance that data whose columns are jointly normal after ranking has a row called far, over all its rows. A row is far when its robust Mahalanobis distance, taken on each column's normal scores (so a skewed or heavy-tailed column does not make rows far by itself), is beyond the chi-squared point that leaves `alpha / rows` above it. (> 0) (< 1) |
| `min_rows_per_column` | `15` | Rows per column below which no row is called far: the spread of so few rows is too loose a ruler. (≥ 2) |
| `min_excess` | `0.05` | How far the forest's top anomaly score must be above the top score it gives a normal sample of the same shape for a row to stand out. Scores vary by 0.01 to 0.02 between samples of that shape (by up to 0.08 now and then at 1 000 rows, so about 5% of clean tables of that size get a low-confidence finding), and a median standing in for missing values lifts them by 0.02. (≥ 0) |
| `n_estimators` | `100` | Trees in the forest. (≥ 10) (≤ 1000) |
| `min_n` | `10` | Finite values a column needs to be a feature. (≥ 4) |
| `max_missing_share` | `0.2` | Share of missing values above which a column is not a feature. A missing value takes the column's median, so a column that is mostly missing makes the rows that have a value look far from the rest. (≥ 0) (≤ 1) |
| `max_rows` | `100000` | Most rows fitted; a larger frame is sampled with `seed`. (≥ 1000) |
| `seed` | `42` | Seed of the forest and of the sample. |

**Output**

`MultivariateOutliersOutput`: `feature_columns`, `rows_used`, `rows_total`, `contamination`, `n_flagged`, `far_rows`, `far_limit`, `example_rows`, `max_anomaly_score`, `reference_max_score`, `excess`

**Finding codes**

`multivariate_anomalies`

## mutual_information

Rank the features by what they tell about the target, linear or not.

Each feature is measured on the rows where it and the target exist, so no value
is imputed. A shuffled copy of the target gives the noise floor of the estimator,
and a feature is notable only above it.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | numeric or categorical or boolean |
| Dimensions | relationship |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | target |
| Needs reference | no |
| Extras | ml |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `target` | `None` | The column to predict. `dsbf.profile(target=...)` sets it; without one the check is not applicable. |
| `n_neighbors` | `3` | Neighbours of the estimator for a continuous feature or target; more smooth the estimate and bias it low. (≥ 1) |
| `min_normalised` | `0.05` | Normalised mutual information from which a feature is reported, when it also exceeds the noise floor. (> 0) (< 1) |
| `min_n` | `30` | Complete rows a feature needs to be measured. (≥ 5) |
| `max_levels` | `50` | Most levels a class column or target may have; the estimate over more is inflated by the levels' own noise. (≥ 2) |
| `max_rows` | `10000` | Most rows read; more are sampled with `seed`. The estimator searches neighbours, so its cost grows faster than the rows. (≥ 1000) |
| `null_draws` | `40` | Estimates against a shuffled target that set a noise floor, the highest of them: a truly independent feature passes it about once in `null_draws + 1`. A class feature gets that many of its own, since its estimate is biased up by its number of levels; the continuous features share one floor. (≥ 12) |
| `null_features` | `6` | Continuous features measured against shuffled targets to set their shared noise floor. (≥ 1) |
| `null_shuffles` | `2` | Shuffled copies of the target each of those features is measured against, or more when that leaves fewer than `null_draws` estimates. (≥ 1) |
| `seed` | `42` | Seed of the row sample, shuffles and estimator. |
| `max_findings` | `20` | Most findings; the most informative first. (≥ 1) |

**Output**

`MutualInformationOutput`: `target`, `target_kind`, `target_classes`, `normalisation`, `features`, `noise_floor`, `features_tested`, `features_notable`, `untested`, `skipped_columns`, `rows_used`, `rows_total`

**Finding codes**

`informative_feature`

## near_zero_variance

Flag numeric columns whose variance is at or below `max_variance`.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | numeric |
| Dimensions | usability |
| Needs profile | counts, numeric_stats |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `max_variance` | `0.0001` | Variance (the square of the standard deviation) at or below which a column is reported. It is in the column's own units, so it does not suit a column measured in very small units. (≥ 0) |

**Output**

`NearZeroVarianceOutput`: `columns`

**Finding codes**

`near_zero_variance`

## normality

Test numeric columns against a normal distribution and keep their Q-Q data.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | numeric |
| Dimensions | distribution |
| Needs profile | counts, numeric_stats |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `alpha` | `0.05` | Chance that a normal column is rejected. Two tests are run, so each is held to half of it. (> 0) (< 1) |
| `min_deviation` | `0.05` | Average distance of the column's quantiles from a normal's, in standard deviations, below which a rejection is not reported. At thousands of rows the tests reject deviations too small to matter: a t distribution with 30 degrees of freedom is 0.02 away, a gamma with skew 0.3 is 0.05. (≥ 0) |
| `min_n` | `8` | Finite values a column needs to be tested. (≥ 4) |
| `shapiro_max_n` | `5000` | Largest sample for Shapiro-Wilk, which rejects trivial deviations above it; larger columns get a Kolmogorov-Smirnov test. (≥ 20) |
| `qq_points` | `100` | Quantile pairs kept for a Q-Q plot. (≥ 10) (≤ 1000) |
| `qq_all_columns` | `False` | Keep Q-Q quantiles for the columns that pass as well as the ones that fail. |

**Output**

`NormalityOutput`: `columns`, `skipped`

**Finding codes**

`non_normal`

## null_rate

Report the share of missing values in each column.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | any type |
| Dimensions | completeness |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `notice_share` | `0.05` | Null share from which a column is reported. (≥ 0) (≤ 1) |
| `warn_share` | `0.2` | Null share from which the finding is a warning. (≥ 0) (≤ 1) |
| `error_share` | `0.5` | Null share from which the finding is an error and the column is listed as mostly missing. (≥ 0) (≤ 1) |

**Output**

`NullRateOutput`: `columns`

**Finding codes**

`high_null_rate`

## numeric_binning

Suggest quantile or equal-width bins, and how many, for each numeric column.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | numeric |
| Dimensions | usability |
| Needs profile | counts, numeric_stats, quantiles |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `skew_threshold` | `1.0` | Absolute skewness from which equal-width bins would crowd most rows into a few of them, so bins of equal row count are suggested. (> 0) |
| `min_core_span` | `0.5` | Share of the range (max − min) that the middle 98% of the values must cover for equal-width bins to be suggested; below it a long tail leaves most bins nearly empty. (> 0) (≤ 1) |
| `min_distinct` | `10` | Distinct values at or below which a column is already discrete and is left alone. (≥ 2) |
| `max_bins` | `10` | Most bins suggested. (≥ 3) |
| `min_rows_per_bin` | `30` | Rows each bin should hold; fewer bins are suggested. (≥ 1) |

**Output**

`NumericBinningOutput`: `columns`, `skipped_columns`

**Finding codes**

`binning_strategy`

## out_of_bounds

Count the values of each bounded numeric column outside its range.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | analysis |
| Cost | cheap |
| Applies to | numeric |
| Dimensions | validity |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `bounds` | `{}` | The range of each column, by name. With none, the check is not applicable: a column's name does not say what range it should hold. |

**Output**

`OutOfBoundsOutput`: `columns`, `unchecked`

**Finding codes**

`out_of_bounds`

## outliers

Flag numeric values beyond the IQR fences, a z-score or a modified z-score.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | numeric |
| Dimensions | distribution |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `methods` | `('iqr', 'zscore', 'mad')` | The rules to apply. Consensus needs at least two of them. |
| `iqr_multiplier` | `1.5` | Tukey's fence distance, in interquartile ranges. (> 0) |
| `zscore_threshold` | `3.0` | \|z\| above which a value is flagged. (> 0) |
| `mad_threshold` | `3.5` | Modified z-score (Iglewicz and Hoaglin, 1993) above which a value is flagged. (> 0) |
| `false_flag_rate` | `0.01` | Chance that a normal column of any length has a value flagged by a rule. The cutoffs of the three rules rise with the number of values until a normal column of n values breaks one with this probability, since the fixed cutoffs (3 deviations, 1.5 interquartile ranges) flag a few values of every long column by chance. They never fall below the fixed cutoffs; 1 uses those alone. (> 0) (≤ 1) |
| `min_n` | `10` | Finite values a column needs to be examined. (≥ 4) |
| `warn_share` | `0.05` | Share of the values one rule flags from which the finding is a warning even without consensus. (> 0) (≤ 1) |

**Output**

`OutliersOutput`: `columns`, `skipped`

**Finding codes**

`outliers_detected`

## quality_dimensions

Report completeness, validity, usability, redundancy and leakage.

| | |
| --- | --- |
| Version | 1.1.1 |
| Kind | synthesis |
| Cost | cheap |
| Applies to | the dataset |
| Dimensions | completeness, validity, usability, redundancy, leakage |
| Needs profile | nothing |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

None.

**Output**

`QualityDimensionsOutput`: `total_columns`, `dimensions`, `clean_columns`

**Finding codes**

None.

## reference_drift

Compare each column's type, nulls and distribution with the reference's.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | any type |
| Dimensions | distribution, validity, completeness |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | yes |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `psi_warn` | `0.1` | Population stability index from which a numeric column has moderate drift. 0.1 and 0.25 are the conventional limits. (> 0) |
| `psi_high` | `0.25` | Index from which the drift is high. (> 0) |
| `tvd_warn` | `0.2` | Total variation distance between the shares of a categorical column's values from which it has moderate drift. (> 0) (≤ 1) |
| `tvd_high` | `0.4` | Distance from which the drift is high. (> 0) (≤ 1) |
| `bins` | `10` | Most bins a numeric distribution is compared on. The reference's histogram is merged into bins that each hold at least this share of it; PSI grows with the number of bins, and its usual limits assume about ten. (≥ 2) (≤ 100) |
| `significance` | `0.01` | Level of the chi-squared test that the two samples are of one distribution (on the PSI for a number, on the counts for a category). A distance below its limits or a test above this level is no drift: a small sample has a large distance by chance, and a huge one a significant difference too small to matter. (> 0) (< 1) |
| `missing_shift` | `0.1` | Change in a column's share of nulls from which it is reported; twice the change is a warning. (> 0) (≤ 1) |
| `target` | `None` | The column to predict. `dsbf.profile(target=...)` sets it; drift in the target is reported one level higher. |
| `target_scale` | `0.5` | The target is held to this share of the PSI and distance limits: a model learns the target itself, and a 15-point move in the share of positives is a total variation distance of 0.15. (> 0) (≤ 1) |

**Output**

`ReferenceDriftOutput`: `reference`, `reference_kind`, `reference_rows`, `reference_rows_profiled`, `rows`, `columns_added`, `columns_dropped`, `columns`, `drifted`

**Finding codes**

`columns_added`, `columns_dropped`, `distribution_shift`, `missingness_shift`, `target_drift`, `target_missing_in_reference`, `type_changed`

## sample_size

Compare the rows with what rules of thumb ask for the number of features.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | any type |
| Dimensions | usability |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `n_features` | `None` | The number of predictors, when only some columns will be used. By default every column except identifiers, datetimes and unknown types. (≥ 1) |
| `linear_rows_per_feature` | `20` | Rows per predictor a linear or logistic regression needs (Harrell 2001; Peduzzi 1996). (≥ 1) |
| `linear_min_rows` | `50` | Rows it needs at least. (≥ 1) |
| `tree_rows_per_feature` | `10` | Rows per predictor a tree-based model needs. (≥ 1) |
| `tree_min_rows` | `100` | Rows it needs at least. (≥ 1) |
| `general_rows_per_feature` | `5` | Rows per predictor for a train, validation, test split. (≥ 1) |
| `general_min_rows` | `1000` | Rows a 60/20/20 split needs for a validation set that gives reliable metrics. (≥ 1) |

**Output**

`SampleSizeOutput`: `rows`, `n_features`, `rows_per_feature`, `verdict`, `rules`

**Finding codes**

`too_few_rows_<...>`

## schema

Compare the columns with the expected types, nulls, ranges and values.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | analysis |
| Cost | cheap |
| Applies to | any type |
| Dimensions | validity |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | columns |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `columns` | `{}` | The expectation for each column, by name. With none, the check is not applicable. |

**Output**

`SchemaOutput`: `checked`, `missing_columns`, `violations`, `valid`

**Finding codes**

`missing_column`

## seasonal_decomposition

Split each numeric column into trend, seasonal and remainder by STL.

The columns are resampled to the frequency first (see ``checks.timeseries``) and
the most recent ``max_periods`` periods are used. The output keeps the strengths,
not the component series.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | numeric or datetime |
| Dimensions | temporal |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | time_index |
| Needs reference | no |
| Extras | timeseries |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `time_index` | `None` | The datetime column that orders the rows. `dsbf.profile(time_index=...)` sets it; without one the check is not applicable. |
| `frequency` | `None` | The spacing of the series, as an offset alias: a count and a unit (`D`, `h`, `min`, `s`, `ms`, `W`: fixed spans; `MS`, `ME`, `QS`, `QE`, `YS`, `YE`: calendar months), such as `15min` or `2W`. Unset, the spacing the profile found in the index is used. |
| `aggregate` | `'mean'` | How the values of one period are combined, which is what repeated timestamps and rows finer than the frequency need. |
| `max_periods` | `2000` | Most periods used: the most recent ones. Tests on tens of thousands of points reject trivial deviations, and the cost of the lag regressions grows with them. (≥ 30) (≤ 100000) |
| `max_fill_share` | `0.2` | Largest share of a series' periods that may be empty and interpolated; a column with more has no series. (≥ 0) (≤ 0.9) |
| `value_columns` | `()` | The numeric columns to analyse; empty is every numeric column. |
| `period` | `None` | Periods in one seasonal cycle. Unset, it follows from the frequency: 7 for daily data, 12 for monthly, 4 for quarterly, 24 for hourly. (≥ 2) (≤ 10000) |
| `robust` | `True` | Fit STL robustly, so outliers do not bend the components. |
| `min_cycles` | `2` | Whole cycles of the period a series needs. (≥ 2) |
| `min_strength` | `0.4` | Smallest trend or seasonal strength that is reported: a moderate one. A series with few cycles or periods must also beat what white noise of its length reaches by chance, which is higher. (≥ 0) (≤ 1) |

**Output**

`SeasonalDecompositionOutput`: `index`, `frequency`, `frequency_source`, `aggregate`, `max_periods`, `periods_total`, `truncated`, `timezone`, `skipped`, `period`, `period_source`, `min_strength`, `columns`

**Finding codes**

`seasonal`, `trend`

## skewness

Flag numeric columns that lean to one side or whose mean sits off the median.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | numeric |
| Dimensions | distribution |
| Needs profile | counts, numeric_stats, quantiles |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `mild` | `0.5` | Absolute skewness above which a column is reported. (> 0) |
| `moderate` | `1.0` | Absolute skewness above which the finding is a warning. (> 0) |
| `heavy` | `2.0` | Absolute skewness above which the skew is called heavy. (> 0) |
| `gap_notice` | `0.35` | Distance between the mean and the median, in standard deviations, from which the gap is reported. A long tail alone puts them up to about a third apart (an exponential, 0.31); a mix of two groups or a column that is mostly zero goes beyond that. (> 0) |
| `gap_warn` | `0.45` | Gap, in standard deviations, from which the finding is a warning. The two can be at most 1 apart; a column that is 55% zero and exponential beyond that has a gap of 0.55, and one that is 70% zero 0.42. (> 0) |
| `gap_min_variance` | `0.0001` | Variance at or below which the gap is not reported: a column that barely varies is `near_zero_variance`'s to report. (≥ 0) |

**Output**

`SkewnessOutput`: `columns`

**Finding codes**

`mean_median_gap`, `skewed`

## stationarity

Test each numeric column for a unit root and a drifting level over time.

The columns are resampled to the frequency first (see ``checks.timeseries``), the
most recent ``max_periods`` periods are used, and the lags are capped.

Limits of the tests: ADF rejects a unit root that is there in about ``alpha`` of
random walks, so about 1 in 20 is read as stationary or as a break; and a short
walk gives KPSS little power, so a random walk of 150 periods is found in about
3 of 4 samples (19 of 20 at 730 periods), the rest being `inconclusive`.

| | |
| --- | --- |
| Version | 1.2.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | numeric or datetime |
| Dimensions | temporal |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | time_index |
| Needs reference | no |
| Extras | timeseries |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `time_index` | `None` | The datetime column that orders the rows. `dsbf.profile(time_index=...)` sets it; without one the check is not applicable. |
| `frequency` | `None` | The spacing of the series, as an offset alias: a count and a unit (`D`, `h`, `min`, `s`, `ms`, `W`: fixed spans; `MS`, `ME`, `QS`, `QE`, `YS`, `YE`: calendar months), such as `15min` or `2W`. Unset, the spacing the profile found in the index is used. |
| `aggregate` | `'mean'` | How the values of one period are combined, which is what repeated timestamps and rows finer than the frequency need. |
| `max_periods` | `2000` | Most periods used: the most recent ones. Tests on tens of thousands of points reject trivial deviations, and the cost of the lag regressions grows with them. (≥ 30) (≤ 100000) |
| `max_fill_share` | `0.2` | Largest share of a series' periods that may be empty and interpolated; a column with more has no series. (≥ 0) (≤ 0.9) |
| `value_columns` | `()` | The numeric columns to analyse; empty is every numeric column. |
| `alpha` | `0.05` | Significance level of both tests. (> 0) (< 1) |
| `max_lag` | `24` | Most lags of the ADF regression, from which AIC picks, and of the KPSS long-run variance. statsmodels' own default grows with the series and every candidate lag is a regression. (≥ 1) (≤ 365) |
| `min_periods` | `20` | Periods a series needs to be tested. (≥ 10) |

**Output**

`StationarityOutput`: `index`, `frequency`, `frequency_source`, `aggregate`, `max_periods`, `periods_total`, `truncated`, `timezone`, `skipped`, `alpha`, `columns`

**Finding codes**

`ambiguous_stationarity`, `non_stationary`, `trend_stationary`

## string_anomalies

Find values that differ only by case, padding, hidden characters or length.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | analysis |
| Cost | cheap |
| Applies to | categorical or text or identifier |
| Dimensions | validity |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `min_values` | `10` | Non-null values a column needs to be read. (≥ 1) |
| `fence` | `3.0` | Multiples of the interquartile range beyond the quartiles of the value lengths from which a length is an outlier. (> 0) |
| `max_examples` | `5` | Examples listed per anomaly. (≥ 1) (≤ 20) |

**Output**

`StringAnomaliesOutput`: `columns`, `columns_read`, `skipped_columns`

**Finding codes**

`edge_whitespace`, `invisible_characters`, `length_outliers`, `mixed_case`

## target_leakage

Flag features that predict the target almost perfectly, or follow it in time.

Four things are looked for, each a reason to ask whether the value exists when
the model predicts: a feature that is nearly a copy of the target, by the
association that suits the pair of types; an identifier or other text with a part
that carries the target; a feature that is present for some outcomes only; and a
datetime column that is later than the time index.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | any type |
| Dimensions | leakage |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | target |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `target` | `None` | The column to predict. `dsbf.profile(target=...)` sets it; without one the check is not applicable. |
| `time_index` | `None` | The datetime column that dates each row's event. `dsbf.profile(time_index=...)` sets it; without one, timestamps are not compared with it. |
| `warn_at` | `0.93` | Association with the target from which a feature is reported as a possible leak. A feature that gets 3 labels in 100 wrong is at 0.94, one that gets 5 in 100 wrong at 0.90, and an honest feature seldom gets within 10. (> 0) (≤ 1) |
| `error_at` | `0.99` | Association from which the finding is an error: the feature is, to within noise, a copy or a recode of the target. (> 0) (≤ 1) |
| `after_share` | `0.95` | Share of the rows in which a datetime column is later than the time index from which it is reported as known only after the event. (> 0) (≤ 1) |
| `min_n` | `30` | Complete rows a feature needs to be measured. (≥ 5) |
| `max_levels` | `50` | Most levels a class column, or a part of an identifier, may have; V over more is inflated by the levels' own noise. (≥ 2) |
| `max_rows` | `200000` | Most rows read; more are sampled with `seed`. (≥ 1000) |
| `seed` | `42` | Seed of the row sample. |
| `max_findings` | `50` | Most findings, strongest first. (≥ 1) |

**Output**

`TargetLeakageOutput`: `target`, `target_kind`, `features`, `features_tested`, `features_flagged`, `untested`, `skipped_columns`, `rows_used`, `rows_total`

**Finding codes**

`derived_from_target`, `missingness_tracks_target`, `target_leakage`, `timestamp_after_event`

## temporal_gaps

Flag missing periods and irregular spacing in datetime columns.

The profile took the dominant step between consecutive distinct values (when it
is at least half of them) and counted the steps longer than 1.5 times it. A
column with no dominant step is irregular: its gaps are not periods missing from
a regular series.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | datetime |
| Dimensions | temporal |
| Needs profile | counts, datetime |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `min_instants` | `10` | Distinct values a column needs before its spacing is judged. (≥ 3) |
| `warn_gaps` | `5` | Gaps from which the finding is a warning rather than information. (≥ 1) |

**Output**

`TemporalGapsOutput`: `columns`, `skipped`

**Finding codes**

`gaps`, `irregular_spacing`

## text_encoding

Report names and values that hold U+FFFD or NUL characters.

U+FFFD is what a lossy decode leaves for bytes it could not read; NUL is what a
UTF-16 file read as UTF-8 leaves between its characters.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | any type |
| Dimensions | validity |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `warn_share` | `0.01` | Share of a column's non-null values above which damaged values are a warning. (≥ 0) (≤ 1) |

**Output**

`TextEncodingOutput`: `columns`

**Finding codes**

`damaged_column_name`, `nul_characters`, `replacement_characters`

## transformations

Try log, root, square, Box–Cox and Yeo–Johnson on each skewed column.

| | |
| --- | --- |
| Version | 1.1.0 |
| Kind | analysis |
| Cost | moderate |
| Applies to | numeric |
| Dimensions | distribution |
| Needs profile | counts, numeric_stats |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `skew_threshold` | `1.0` | Absolute skewness from which a column is previewed. (> 0) |
| `target_skew` | `0.5` | Absolute skewness after the transform at which the simplest transform that gets there is suggested. (≥ 0) |
| `min_reduction` | `0.25` | Share of the absolute skewness a transform must remove to be suggested when none reaches the target, and it must also leave the column under `skew_threshold`. (> 0) (≤ 1) |
| `min_n` | `10` | Finite values a column needs. (≥ 4) |
| `max_rows` | `20000` | Most values transformed; a larger column is sampled with `seed`. (≥ 100) |
| `seed` | `42` | Seed of the row sample. |

**Output**

`TransformationsOutput`: `columns`, `skipped_columns`

**Finding codes**

`no_effective_transform`, `transform_suggested`

## zeros

Flag numeric columns in which zero is a large share of the values.

| | |
| --- | --- |
| Version | 1.0.0 |
| Kind | rule |
| Cost | cheap |
| Applies to | numeric |
| Dimensions | distribution |
| Needs profile | counts |
| Needs checks | none |
| Needs parameters | none |
| Needs reference | no |
| Extras | none |
| Packs | core |

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `notice_share` | `0.3` | Zero share from which a column is reported. (≥ 0) (≤ 1) |
| `warn_share` | `0.5` | Zero share from which the finding is a warning. (≥ 0) (≤ 1) |
| `flag_share` | `0.95` | Zero share above which a column is flagged. (≥ 0) (≤ 1) |

**Output**

`ZerosOutput`: `columns`

**Finding codes**

`high_zero_rate`
