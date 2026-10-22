"""The differences between the profile and the legacy outputs that DSBF accepts.

Each entry has a slug, which ``docs/parity/profile.md`` also uses as the heading of its
reason. ``cells`` lists the golden ``(dataset, column)`` pairs where the difference
shows up and a parity test must expect it; an entry with none is a rule the tests build
in (a different default, a renamed type) or a difference no golden dataset reaches.
``test_parity_report`` keeps the document and this list in step.
"""

from dataclasses import dataclass
from typing import Literal

Area = Literal["load", "types", "columns", "values", "temporal", "dataset", "stage"]


@dataclass(frozen=True)
class Accepted:
    slug: str
    area: Area
    legacy: str
    new: str
    reason: str
    cells: frozenset[tuple[str, str]] = frozenset()


def _entry(
    slug: str,
    area: Area,
    legacy: str,
    new: str,
    reason: str,
    *cells: tuple[str, str],
) -> Accepted:
    return Accepted(slug, area, legacy, new, reason, frozenset(cells))


ACCEPTED_DIFFERENCES: tuple[Accepted, ...] = (
    # --- load: `load` against `pd.read_csv` / `pd.read_excel` ---
    _entry(
        "all-null-column-is-string",
        "load",
        "float64",
        "String",
        "A CSV column with no values has no type to read. pandas calls it float64; "
        "Polars reads text. The profile types it `unknown` either way.",
    ),
    _entry(
        "empty-csv-header-keeps-its-name",
        "load",
        "Unnamed: 6",
        "''",
        "Polars keeps an empty CSV header as the empty string; pandas invents a "
        "name from the position.",
    ),
    _entry(
        "empty-excel-header-is-named-by-position",
        "load",
        "Unnamed: 6",
        "__UNNAMED__6",
        "The Excel reader (fastexcel) names an empty header by its position.",
    ),
    _entry(
        "midnight-excel-datetimes-are-dates",
        "load",
        "datetime64",
        "Date",
        "An Excel column whose datetimes are all at midnight reads as Date.",
    ),
    # --- types: `infer_column_types` against `infer_types` ---
    _entry(
        "semantic-type-names",
        "types",
        "continuous, id",
        "numeric, identifier",
        "Two types were renamed in the shared vocabulary. The comparison maps the "
        "legacy names before it compares.",
    ),
    _entry(
        "boolean-column-type",
        "types",
        "categorical",
        "boolean",
        "A Polars Boolean column is `boolean`. The legacy task saw a pandas bool "
        "column and called it categorical.",
    ),
    _entry(
        "datetime-past-pandas-range",
        "types",
        "identifier",
        "datetime",
        "The column's years run to 2259, past pandas' nanosecond limit. The legacy "
        "parse reads 89% of it and misses its 90% line; Polars holds the dates.",
        ("kitchen5k", "datetime_with_gaps"),
    ),
    _entry(
        "prose-is-text-not-a-key",
        "types",
        "identifier",
        "text",
        "Several-word strings longer than 30 characters are free text even when "
        "nearly every value is distinct. The legacy task tested the distinct share "
        "first and called prose a key; `id_columns` and `high_cardinality` would "
        "then have flagged it.",
        ("kitchen5k", "long_text"),
        ("sms_spam", "text"),
    ),
    _entry(
        "fixed-date-format-list",
        "types",
        "any format dateutil parses",
        "one of 19 listed formats",
        "A fixed list replaces the dateutil fallback, which cost 8.6 s on kitchen5k. "
        "A format outside the list is not a date. `%Y%m%d` is left out because "
        "eight-digit identifiers parse under it.",
    ),
    # --- columns: counts and numeric statistics ---
    _entry(
        "categorical-integer-codes-have-no-statistics",
        "columns",
        "min, max, mean, std and percentiles",
        "top values only",
        "The legacy `summarize_numeric` summarises every numeric dtype. The profile "
        "gives numeric statistics to `numeric` columns, so integer codes typed "
        "`categorical` carry top values instead. Adding min and max to a categorical "
        "profile would change the published schema for one legacy consumer.",
        ("titanic", "pclass"),
        ("titanic", "sibsp"),
        ("titanic", "parch"),
        ("titanic", "survived"),
        ("kitchen5k", "balanced_target"),
        ("kitchen5k", "constant_numeric"),
        ("kitchen5k", "imbalanced_target"),
        ("kitchen5k", "mostly_zeros"),
        ("kitchen5k", "multiclass_target"),
        ("kitchen5k", "near_zero_variance"),
        ("kitchen5k", "perfect_leakage"),
        ("kitchen5k", "target_for_leakage"),
    ),
    # --- values: top values and lengths ---
    _entry(
        "top-k-default",
        "values",
        "5 values",
        "10 values",
        "The profile lists ten top values by default. The comparison asks for five.",
    ),
    _entry(
        "null-is-not-a-top-value",
        "values",
        "null counted as a value",
        "non-null values only",
        "The legacy table is `value_counts(dropna=False)`. The profile's top values "
        "are of the non-null values, whose count is `n_non_null`.",
    ),
    _entry(
        "text-lists-repeated-values-only",
        "values",
        "every value",
        "values that repeat",
        "In a text column nearly every value is unique, and a list of them is not a "
        "profile.",
    ),
    _entry(
        "long-strings-are-not-top-values",
        "values",
        "any length",
        "at most 256 characters",
        "One long column made a 12.5 KB profile entry.",
    ),
    _entry(
        "average-word-length",
        "values",
        "all characters per word",
        "non-space characters per word",
        "The legacy mean divides every character, spaces included, by the words.",
    ),
    _entry(
        "csv-float-last-digit",
        "values",
        "pandas' float parser",
        "Polars' float parser",
        "pandas' default CSV parser is not correctly rounded, so a value can differ "
        "in the last digit. Floats are compared to a relative 1e-12.",
    ),
    _entry(
        "nul-byte-cuts-a-pandas-field",
        "values",
        "field cut at a NUL byte",
        "whole value",
        "pandas' C parser cuts a field at a NUL byte (`text\\x00with\\x1bcontrols` "
        "reads as `text`), so its values and lengths are shorter.",
        ("kitchen5k", "control_character_text"),
    ),
    # --- temporal: datetime profiles against `pd.infer_freq` ---
    _entry(
        "frequency-names-the-spacing",
        "temporal",
        "anchored alias (`W-SUN`, `MS`)",
        "spacing alias (`W`, `MS`/`ME`)",
        "A weekly series is any weekday and a monthly one any month position; the "
        "profile does not report the anchor.",
    ),
    _entry(
        "jittered-timestamps-have-no-frequency",
        "temporal",
        "no frequency",
        "no frequency and no gap count",
        "A series with no dominant step has neither a frequency nor gaps.",
    ),
    # --- dataset: shape, duplicates and nulls ---
    _entry(
        "null-cell-share-is-a-fraction",
        "dataset",
        "`null_cell_percentage`, a fraction",
        "`rows_with_any_null` and per-column null counts",
        "The legacy value is named a percentage and holds a fraction (0.0592 for "
        "5.92%). The profile reports counts; the comparison derives the fraction.",
    ),
    # --- stage: `infer_stage` against the legacy `infer_stage` ---
    _entry(
        "stage-names",
        "stage",
        "model_ready, exploratory",
        "modeling, raw",
        "`model_ready` is `modeling` and `exploratory` is `raw`: the data has not "
        "been prepared. `all_categorical`, `high_missingness`, `wide` and "
        "`kitchen5k` differ only in this name.",
    ),
    _entry(
        "high-cardinality-counts-categorical-columns-only",
        "stage",
        "every column with over 50 distinct values",
        "categorical columns with over 50 distinct values",
        "Counting continuous and identifier columns kept `clean`, `near_clean`, "
        "`severe_multicollinearity`, `sms_spam` and `titanic` from being `cleaned`, "
        "though none has a high-cardinality categorical column. They are `cleaned` "
        "now.",
    ),
    _entry(
        "stage-inference-honours-the-configured-thresholds",
        "stage",
        "defaults, whatever the config says",
        "`engine.stage_inference`",
        "The legacy function read its thresholds from the wrong config path.",
    ),
    _entry(
        "stage-does-not-depend-on-the-data-library",
        "stage",
        "pandas and Polars input differ",
        "one answer",
        "The legacy function compared Polars dtypes to strings, so the same data "
        "was `model_ready` from pandas and `exploratory` from Polars "
        "(`single_column`). The new function reads only the profile.",
    ),
)

BY_SLUG: dict[str, Accepted] = {a.slug: a for a in ACCEPTED_DIFFERENCES}
assert len(BY_SLUG) == len(ACCEPTED_DIFFERENCES), "duplicate slug"


def cells(slug: str) -> frozenset[tuple[str, str]]:
    """The golden ``(dataset, column)`` pairs where this difference shows."""
    return BY_SLUG[slug].cells
