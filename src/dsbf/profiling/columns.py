# dsbf/profiling/columns.py
"""Per-column counts, statistics, histograms, top values, text and datetime profiles.

Every count and numeric aggregate for every column goes into a single Polars
``select``, so the engine reads the frame once and works on the columns in parallel.
Each statistic is over that column's own values, never over rows that are complete
across columns (a column with nulls does not shrink the others). The profile parts
that need the values themselves (histogram bins, top values, string shape, datetime
spacing) are then built per column by ``histogram``, ``values`` and ``temporal``.

Nulls are counted apart from ``non_finite`` values: a pandas ``NaN`` is null after
``load``, while a ``NaN`` or infinity in a Polars float column is a value, counted in
``non_finite`` and left out of the numeric statistics. Those statistics are pandas'
own definitions: sample standard deviation, bias-corrected skewness and excess
kurtosis, and linear-interpolation quantiles.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import polars as pl

from dsbf.data.dataset import Dataset
from dsbf.model.profile import (
    BooleanProfile,
    CategoricalProfile,
    ColumnCounts,
    ColumnProfile,
    DatetimeProfile,
    IdentifierProfile,
    NumericProfile,
    NumericStats,
    Quantile,
    TextProfile,
    UnknownProfile,
)
from dsbf.profiling.histogram import DEFAULT_MAX_BINS, histogram
from dsbf.profiling.temporal import datetime_stats
from dsbf.profiling.types import ColumnType, infer_column_types
from dsbf.profiling.values import DEFAULT_TOP_K, string_stats, top_values, value_counts

DEFAULT_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
_TEXT_DTYPES = (pl.String, pl.Categorical, pl.Enum)


def profile_columns(
    dataset: Dataset,
    types: Mapping[str, ColumnType] | None = None,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    top_k: int = DEFAULT_TOP_K,
    max_bins: int = DEFAULT_MAX_BINS,
) -> tuple[ColumnProfile, ...]:
    """The profile of every column: counts, then what its semantic type calls for.

    Numeric columns get statistics and an equal-width histogram of at most
    ``max_bins`` bins. Categorical and text columns get their ``top_k`` most frequent
    values (text lists only values that repeat), and string columns also their
    lengths, character classes and ``top_k`` pattern signatures. Datetime columns get
    their range, time zone, frequency and gaps. ``top_k=0`` leaves out top values and
    patterns.

    ``types`` are the columns' semantic types; they are inferred when not given, so a
    caller that already has them passes them in rather than inferring twice. The
    profiles come back in column order.

    Raises:
        ValueError: ``types`` does not cover exactly the dataset's columns, names a
            numeric type for a column that is not numeric, ``quantiles`` are not
            increasing values in [0, 1], ``top_k`` is negative or ``max_bins`` is
            below 1.
    """
    quantiles = _check_quantiles(quantiles)
    if top_k < 0:
        raise ValueError(f"top_k must not be negative, not {top_k!r}")
    if max_bins < 1:
        raise ValueError(f"max_bins must be at least 1, not {max_bins!r}")
    types = infer_column_types(dataset) if types is None else types
    frame = dataset.frame
    if set(types) != set(frame.columns):
        raise ValueError("types must name exactly the dataset's columns")
    for name, column_type in types.items():
        dtype = frame.schema[name]
        if column_type.semantic_type == "numeric" and not dtype.is_numeric():
            raise ValueError(f"{name!r} is typed numeric but holds {dtype}")

    plan = [_expressions(i, frame, types, quantiles) for i in range(frame.width)]
    row = frame.select([e for exprs in plan for e in exprs]).row(0, named=True)
    return tuple(
        _profile(i, dataset, types[name], row, quantiles, top_k, max_bins)
        for i, name in enumerate(frame.columns)
    )


def _check_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    values = tuple(quantiles)
    if not all(0 <= p <= 1 for p in values) or any(
        a >= b for a, b in zip(values, values[1:], strict=False)
    ):
        raise ValueError("quantiles must increase and lie in [0, 1]")
    return values


def _key(position: int, stat: str) -> str:
    # Positions key the one-row result, so two columns can never share an alias.
    return f"{position}:{stat}"


def _expressions(
    position: int,
    frame: pl.DataFrame,
    types: Mapping[str, ColumnType],
    quantiles: tuple[float, ...],
) -> list[pl.Expr]:
    name = frame.columns[position]
    dtype = frame.schema[name]
    column = pl.col(name)
    exprs = [column.null_count().alias(_key(position, "nulls"))]
    if dtype != pl.Object:
        exprs.append(column.drop_nulls().n_unique().alias(_key(position, "distinct")))
    semantic_type = types[name].semantic_type
    if semantic_type == "numeric":
        exprs.extend(_numeric_expressions(position, column, quantiles))
    elif semantic_type == "boolean" and dtype == pl.Boolean:
        exprs.append(column.sum().alias(_key(position, "true")))
    elif semantic_type in ("categorical", "text") and dtype in _TEXT_DTYPES:
        empty = (column.cast(pl.String) == "").sum()
        exprs.append(empty.alias(_key(position, "empty")))
    return exprs


def _numeric_expressions(
    position: int, column: pl.Expr, quantiles: tuple[float, ...]
) -> list[pl.Expr]:
    # Moments are taken in Float64 so UInt64, Int128 and Decimal cannot overflow.
    as_float = column.cast(pl.Float64)
    finite = as_float.is_finite()
    values = as_float.filter(finite)
    stats = {
        "zeros": (column == 0).sum(),
        "non_finite": (column.is_not_null() & ~finite).sum(),
        "finite": finite.sum(),
        # Min and max in the stored dtype, so an integer beyond 2**53 stays exact
        # until it is cast.
        "min": column.filter(finite).min().cast(pl.Float64),
        "max": column.filter(finite).max().cast(pl.Float64),
        "mean": values.mean(),
        "std": values.std(ddof=1),
        "skewness": values.skew(bias=False),
        "kurtosis": values.kurtosis(bias=False),
    }
    exprs = [expr.alias(_key(position, stat)) for stat, expr in stats.items()]
    exprs.extend(
        values.quantile(p, "linear").alias(_key(position, f"q{index}"))
        for index, p in enumerate(quantiles)
    )
    return exprs


def _profile(
    position: int,
    dataset: Dataset,
    column_type: ColumnType,
    row: dict[str, Any],
    quantiles: tuple[float, ...],
    top_k: int,
    max_bins: int,
) -> ColumnProfile:
    name = dataset.columns[position]
    series = dataset.frame.get_column(name)
    nulls = row[_key(position, "nulls")]
    if series.dtype == pl.Object:
        distinct = _object_distinct(series)
    else:
        distinct = row[_key(position, "distinct")]
    common: dict[str, Any] = {
        "name": name,
        "position": position,
        "physical_dtype": str(series.dtype),
        "semantic_confidence": column_type.confidence,
        "semantic_sample_rows": column_type.sample_rows,
        "counts": ColumnCounts(rows=dataset.n_rows, nulls=nulls, distinct=distinct),
    }
    match column_type.semantic_type:
        case "numeric":
            return _numeric_profile(position, common, row, quantiles, series, max_bins)
        case "boolean":
            true_count = row.get(_key(position, "true"))
            if true_count is None:
                return UnknownProfile(**common)
            return BooleanProfile(
                **common,
                true_count=true_count,
                false_count=dataset.n_rows - nulls - true_count,
            )
        case "categorical":
            return CategoricalProfile(
                **common,
                empty_strings=row.get(_key(position, "empty"), 0),
                **_value_fields(series, top_k, text=False),
            )
        case "text":
            return TextProfile(
                **common,
                empty_strings=row.get(_key(position, "empty"), 0),
                **_value_fields(series, top_k, text=True),
            )
        case "datetime":
            return DatetimeProfile(
                **common, **asdict(datetime_stats(series, column_type))
            )
        case "identifier":
            return IdentifierProfile(**common)
        case _:
            return UnknownProfile(**common)


def _numeric_profile(
    position: int,
    common: dict[str, Any],
    row: dict[str, Any],
    quantiles: tuple[float, ...],
    series: pl.Series,
    max_bins: int,
) -> NumericProfile:
    def stat(name: str) -> Any:
        return row[_key(position, name)]

    finite = stat("finite")
    stats = None
    if finite:
        # Polars returns null where pandas returns NaN (the spread of one value,
        # the skewness of two); NaN lets the contract record it as undefined.
        def moment(name: str) -> float:
            return math.nan if stat(name) is None else stat(name)

        stats = NumericStats(
            min=stat("min"),
            max=stat("max"),
            mean=stat("mean"),
            std=moment("std"),
            skewness=moment("skewness"),
            kurtosis=moment("kurtosis"),
            quantiles=tuple(
                Quantile(p=p, value=stat(f"q{index}"))
                for index, p in enumerate(quantiles)
            ),
            quantile_method="exact" if quantiles else None,
        )
    bins = None
    if stats is not None:
        bins = histogram(series, max_bins=max_bins, quartiles=_quartiles(stats))
    return NumericProfile(
        **common,
        zeros=stat("zeros"),
        non_finite=stat("non_finite"),
        stats=stats,
        histogram=bins,
    )


def _quartiles(stats: NumericStats) -> tuple[float, float] | None:
    """The 25th and 75th percentiles, when the profile computed both."""
    by_p = {q.p: q.value for q in stats.quantiles}
    if 0.25 in by_p and 0.75 in by_p:
        return by_p[0.25], by_p[0.75]
    return None


def _value_fields(series: pl.Series, top_k: int, *, text: bool) -> dict[str, Any]:
    """Top values and, for strings, lengths and shape of a categorical or text column.

    A text column lists only values that repeat: the first-seen of many one-off
    strings is not a top value.
    """
    counts = value_counts(series)
    fields: dict[str, Any] = {
        "top_values": top_values(counts, top_k, min_count=2 if text else 1)
    }
    if series.dtype in _TEXT_DTYPES:
        stats = string_stats(counts, top_k)
        fields["lengths"] = stats.lengths
        fields["shape"] = stats.shape
        if text:
            fields["avg_word_length"] = stats.avg_word_length
    return fields


def _object_distinct(series: pl.Series) -> int:
    """Distinct non-null values of an Object column, which Polars cannot count."""
    values = [value for value in series.to_list() if value is not None]
    try:
        return len(set(values))
    except TypeError:
        return len({repr(value) for value in values})
