# dsbf/profile/columns.py
"""Per-column counts and numeric statistics, computed in one pass.

Every aggregate for every column goes into a single Polars ``select``, so the engine
reads the frame once and works on the columns in parallel. Each statistic is over
that column's own values, never over rows that are complete across columns (a column
with nulls does not shrink the others).

Nulls are counted apart from ``non_finite`` values: a pandas ``NaN`` is null after
``load``, while a ``NaN`` or infinity in a Polars float column is a value, counted in
``non_finite`` and left out of the numeric statistics. Those statistics are pandas'
own definitions: sample standard deviation, bias-corrected skewness and excess
kurtosis, and linear-interpolation quantiles.
"""

import math
from collections.abc import Mapping, Sequence
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
from dsbf.profile.types import ColumnType, infer_column_types

DEFAULT_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
_TEXT_DTYPES = (pl.String, pl.Categorical, pl.Enum)


def profile_columns(
    dataset: Dataset,
    types: Mapping[str, ColumnType] | None = None,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> tuple[ColumnProfile, ...]:
    """The counts and, for numeric columns, the statistics of every column.

    ``types`` are the columns' semantic types; they are inferred when not given, so a
    caller that already has them passes them in rather than inferring twice. The
    profiles come back in column order. Histograms, top values, lengths and datetime
    ranges are not part of this pass.

    Raises:
        ValueError: ``types`` does not cover exactly the dataset's columns, names a
            numeric type for a column that is not numeric, or ``quantiles`` are not
            increasing values in [0, 1].
    """
    quantiles = _check_quantiles(quantiles)
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
        _profile(i, dataset, types[name], row, quantiles)
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
            return _numeric_profile(position, common, row, quantiles)
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
                **common, empty_strings=row.get(_key(position, "empty"), 0)
            )
        case "text":
            return TextProfile(
                **common, empty_strings=row.get(_key(position, "empty"), 0)
            )
        case "datetime":
            return DatetimeProfile(**common)
        case "identifier":
            return IdentifierProfile(**common)
        case _:
            return UnknownProfile(**common)


def _numeric_profile(
    position: int,
    common: dict[str, Any],
    row: dict[str, Any],
    quantiles: tuple[float, ...],
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
    return NumericProfile(
        **common,
        zeros=stat("zeros"),
        non_finite=stat("non_finite"),
        stats=stats,
    )


def _object_distinct(series: pl.Series) -> int:
    """Distinct non-null values of an Object column, which Polars cannot count."""
    values = [value for value in series.to_list() if value is not None]
    try:
        return len(set(values))
    except TypeError:
        return len({repr(value) for value in values})
