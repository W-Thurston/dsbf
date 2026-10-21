# Profile parity with the legacy outputs

The profile (`dsbf.profiling`) replaces the per-task scans of the legacy `summarize_*`,
`detect_duplicates` and `infer_types` tasks and the legacy `infer_stage`. This page lists
where the two disagree on the 10 golden datasets and why each difference is accepted.
Anything not listed here is a failure: the parity tests compare the two and fail on a
difference they don't expect.

The tests are in `tests/test_profile/`; the accepted differences are in
`accepted.py` there, and `test_parity_report.py` fails if this page and that list drift
apart. Each heading below is a slug from that list.

| Area | Legacy output | Test | Result |
|---|---|---|---|
| Load | `pd.read_csv`: shape, names, null counts | `test_load_parity.py` | Equal on all 10 datasets |
| Types | `infer_types` | `test_type_parity.py` | Equal except the types below |
| Columns | `summarize_nulls`, `summarize_unique`, `summarize_numeric` | `test_column_parity.py` | Equal within `rel=1e-9`, except integer codes |
| Values | `summarize_value_counts`, `categorical_length_stats` | `test_column_parity.py` | Equal except the rules below |
| Dataset | `summarize_dataset_shape`, `detect_duplicates` | `test_dataset_parity.py` | Equal |
| Entropy | `compute_entropy` | `test_rule_inputs_parity.py` | Equal on the same columns, within the six significant figures the snapshot keeps |
| Duplicate columns | `detect_duplicate_columns` | `test_rule_inputs_parity.py` | Equal: a pair is equal hashes with equal dtypes |
| Stage | `infer_stage` (snapshot `inferred_stage`) | `test_dataset_parity.py` | 9 of 10 differ, all listed |

The golden datasets are `clean`, `tiny`, `near_clean`, `all_categorical`,
`high_missingness`, `severe_multicollinearity`, `single_column`, `wide`, `titanic` and
`kitchen5k` (see `tests/golden/README.md`).

## Not compared

- **Missingness co-occurrence.** No legacy task reports which columns are null together.
  `missingness_heatmap` reports totals and `missingness_mechanism_analysis` runs tests.
  The profile's pairs are checked against a pandas cross-tabulation in `test_dataset.py`.
- **Histograms and string shape.** The legacy plots and text tasks keep no comparable
  numbers: `summarize_text_fields` returns nothing on any golden dataset. Histogram counts
  are checked against `np.histogram` in `test_histogram.py`.
- **Encoding damage.** No legacy task counts U+FFFD or NUL. The counts, and the flag on a
  damaged column name, are checked on built frames and on a UTF-16 CSV read with
  `utf8-lossy` in `test_columns.py`. `load` itself refuses a UTF-16 or Latin-1 CSV (polars
  2.0.0: `invalid utf-8 sequence`), so the counts see U+FFFD and NUL that arrive in
  Parquet, JSON, Excel and in-memory frames, or inside a UTF-8 CSV.
- **Datetime profiles.** The legacy tasks report no frequency or range. The profile is
  checked against `pd.infer_freq` in `test_temporal.py`.

## Load (`load`)

#### `all-null-column-is-string`

A CSV column with no values has no type to read. pandas calls it `float64` and Polars
reads text. The profile types it `unknown` either way.

#### `empty-csv-header-keeps-its-name`

Polars keeps an empty CSV header as the empty string. pandas invents `Unnamed: 6` from the
position.

#### `empty-excel-header-is-named-by-position`

The Excel reader (fastexcel) names an empty header `__UNNAMED__6`. `pd.read_excel` needs
`openpyxl`, which is not a dependency, so this is pinned on `load` alone.

#### `midnight-excel-datetimes-are-dates`

An Excel column whose datetimes are all at midnight reads as `Date`.

## Semantic types (`types`)

#### `semantic-type-names`

Legacy `continuous` is `numeric` and `id` is `identifier`. The comparison maps the names
before it compares.

#### `boolean-column-type`

A Polars `Boolean` column is `boolean`. The legacy task saw a pandas bool column and
called it `categorical`.

#### `datetime-past-pandas-range`

Kitchen5k's `datetime_with_gaps` has years up to 2259, past pandas' nanosecond limit. The
legacy parse reads 89% of it and misses its 90% line, so legacy said `identifier`.

#### `fixed-date-format-list`

The legacy dateutil fallback is replaced by a list of 19 formats, which cut inference on
kitchen5k from 25.95 s to 0.3 s. A format outside the list is not a date. `%Y%m%d` is left
out because eight-digit identifiers parse under it. No golden dataset reaches this.

## Counts and numeric statistics (`columns`)

Null counts, distinct counts, and the count, min, max, mean, std and seven percentiles of
every numeric column equal the legacy tasks within `rel=1e-9` on all 10 datasets, with one
exception.

#### `categorical-integer-codes-have-no-statistics`

The legacy `summarize_numeric` summarises every numeric dtype. The profile gives numeric
statistics to `numeric` columns, so integer codes typed `categorical` carry top values
instead: titanic's `pclass`, `sibsp`, `parch` and `survived`, and eight kitchen5k target
and flag columns. The test asserts each is a categorical profile, so the entry cannot go
stale.

Decision: a categorical profile does not carry min and max. It would change the published
schema for one legacy consumer, and the check that reads these columns moves onto the
profile in a later phase, where the type it should read is decided.

## Top values and lengths (`values`)

#### `top-k-default`

The profile lists ten top values by default; the legacy task shows five. The comparison
asks for five.

#### `null-is-not-a-top-value`

The legacy table is `value_counts(dropna=False)`, so null is a value. The profile's top
values are of the non-null values.

#### `text-lists-repeated-values-only`

A text column lists only values that repeat. Nearly every value in text is unique, and a
list of them is not a profile.

#### `long-strings-are-not-top-values`

No string over 256 characters is listed. One such column made a 12.5 KB entry.

#### `average-word-length`

The profile divides non-space characters by words. The legacy `avg_word_length` divides all
characters, spaces included. No golden dataset has a legacy text summary to compare, so the
rule is covered by `test_values.py`.

#### `csv-float-last-digit`

pandas' default CSV float parser is not correctly rounded, so a value such as
`100.00020190141393` can read one unit in the last place from Polars' reading. Floats are
compared to a relative `1e-12`.

#### `nul-byte-cuts-a-pandas-field`

pandas' C parser cuts a field at a NUL byte, so kitchen5k's `control_character_text`
(`text\x00with\x1bcontrols` reads as `text`) has shorter values and lengths. The profile
keeps the whole value.

## Datetime (`temporal`)

Compared with `pd.infer_freq` on regular series in `test_temporal.py`.

#### `frequency-names-the-spacing`

The profile's frequency names the spacing, not the anchor: `W` is any weekday and
`MS`/`ME` are any month position. `pd.infer_freq` returns `W-SUN` or `MS`.

#### `jittered-timestamps-have-no-frequency`

A series with no dominant step gets no frequency and no gap count.

## Dataset (`dataset`)

Shape, duplicate rows and null cells equal `summarize_dataset_shape` and
`detect_duplicates` on all 10 datasets.

#### `null-cell-share-is-a-fraction`

The legacy `null_cell_percentage` holds a fraction (0.0592 for 5.92%). The profile reports
counts of nulls per column and rows with any null; the comparison derives the fraction and
rounds it to four places, as the legacy task does.

## Stage (`stage`)

Nine of ten golden datasets get a different name from the legacy function. Five are the same
answer in the new vocabulary (the first four rows and `single_column`) and four are a changed
answer. `tiny` (`cleaned`) is the only dataset with the same name.

| Dataset | Legacy | Profile | Why |
|---|---|---|---|
| `all_categorical` | `exploratory` | `raw` | `stage-names`; a high-cardinality categorical column (`tag`) |
| `high_missingness` | `exploratory` | `raw` | `stage-names`; 21% of cells null, over the 10% a `cleaned` stage allows |
| `wide` | `exploratory` | `raw` | `stage-names`; a high-cardinality categorical column (`cat_hc`) |
| `kitchen5k` | `exploratory` | `raw` | `stage-names`; three high-cardinality categorical columns |
| `clean` | `exploratory` | `cleaned` | `high-cardinality-counts-categorical-columns-only` |
| `near_clean` | `exploratory` | `cleaned` | `high-cardinality-counts-categorical-columns-only` |
| `severe_multicollinearity` | `exploratory` | `cleaned` | `high-cardinality-counts-categorical-columns-only` |
| `titanic` | `exploratory` | `cleaned` | `high-cardinality-counts-categorical-columns-only` |
| `single_column` | `model_ready` | `modeling` | `stage-names` |
| `tiny` | `cleaned` | `cleaned` | none |

#### `stage-names`

`model_ready` is `modeling` and `exploratory` is `raw`: the data has not been prepared.

#### `high-cardinality-counts-categorical-columns-only`

The legacy function counted every column with over 50 distinct values, so continuous and
identifier columns kept `clean`, `near_clean`, `severe_multicollinearity` and `titanic`
from being `cleaned`. They have no high-cardinality categorical column and are `cleaned`.

#### `stage-inference-honours-the-configured-thresholds`

The legacy function read its thresholds from the wrong config path and always used the
defaults. The profile's reads `engine.stage_inference`.

#### `stage-does-not-depend-on-the-data-library`

The legacy function compared Polars dtypes to strings, so the same `single_column` data was
`model_ready` from pandas and `exploratory` from Polars. The new function reads only the
profile; `test_stage.py` holds the regression test.
