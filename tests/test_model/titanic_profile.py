"""A profile of the vendored titanic data, computed here with pandas and NumPy.

This stands in for the profiling pass (Phase 2) so the contracts are exercised on
real values and the size budget is measured on a realistic profile. It is not the
profiler: the semantic types come from a few blunt rules.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from dsbf.model import (
    BooleanProfile,
    CategoricalProfile,
    ColumnCounts,
    DatasetProfile,
    Histogram,
    LengthStats,
    NumericProfile,
    NumericStats,
    Quantile,
    ValueCount,
)

CSV = Path(__file__).parents[1] / "golden" / "data" / "titanic.csv"
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
TOP_K = 10


def load() -> pd.DataFrame:
    return pd.read_csv(CSV)


def _counts(series: pd.Series) -> ColumnCounts:
    return ColumnCounts(
        rows=len(series), nulls=int(series.isna().sum()), distinct=series.nunique()
    )


def _top_values(values: pd.Series) -> tuple[ValueCount, ...]:
    top = values.value_counts().head(TOP_K)
    return tuple(
        ValueCount(value=cast(Any, v.item() if hasattr(v, "item") else v), count=int(n))
        for v, n in top.items()
    )


def _numeric(series: pd.Series, position: int) -> NumericProfile:
    values = series.dropna().to_numpy(dtype=float)
    edges = np.histogram_bin_edges(values, bins="fd")
    counts, _ = np.histogram(values, bins=edges)
    return NumericProfile(
        name=str(series.name),
        position=position,
        physical_dtype=str(series.dtype),
        counts=_counts(series),
        zeros=int((values == 0).sum()),
        stats=NumericStats(
            min=values.min(),
            max=values.max(),
            mean=values.mean(),
            std=values.std(ddof=1),
            skewness=cast(float, pd.Series(values).skew()),
            kurtosis=cast(float, pd.Series(values).kurt()),
            quantiles=tuple(
                Quantile(p=q, value=float(np.quantile(values, q))) for q in QUANTILES
            ),
            quantile_method="exact",
        ),
        histogram=Histogram(
            edges=tuple(edges.tolist()), counts=tuple(counts.tolist()), rule="fd"
        ),
    )


def _column(series: pd.Series, position: int):
    values = series.dropna()
    if series.dtype == bool:
        return BooleanProfile(
            name=str(series.name),
            position=position,
            physical_dtype="Boolean",
            counts=_counts(series),
            true_count=int(values.sum()),
            false_count=int((~values).sum()),
        )
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
        return _numeric(series, position)
    lengths = values.astype(str).str.len()
    return CategoricalProfile(
        name=str(series.name),
        position=position,
        physical_dtype=str(series.dtype),
        counts=_counts(series),
        top_values=_top_values(values),
        lengths=LengthStats(
            min=int(lengths.min()), max=int(lengths.max()), mean=float(lengths.mean())
        ),
    )


def build(frame: pd.DataFrame | None = None) -> DatasetProfile:
    frame = load() if frame is None else frame
    return DatasetProfile(
        n_rows=len(frame),
        approx_memory_bytes=int(frame.memory_usage(deep=True).sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        rows_with_any_null=int(frame.isna().any(axis=1).sum()),
        columns=tuple(
            _column(frame[name], position) for position, name in enumerate(frame)
        ),
    )
