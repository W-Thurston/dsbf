# dsbf/profiling/dataset.py
"""The dataset-level profile: shape, duplicate rows and missingness co-occurrence."""

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from dsbf.data.dataset import Dataset
from dsbf.model.profile import (
    ColumnProfile,
    DatasetProfile,
    Missingness,
    NullPair,
)
from dsbf.profiling.columns import DEFAULT_QUANTILES, profile_columns
from dsbf.profiling.histogram import DEFAULT_MAX_BINS
from dsbf.profiling.types import ColumnType, infer_column_types
from dsbf.profiling.values import DEFAULT_TOP_K

DEFAULT_MAX_PAIR_COLUMNS = 100
DEFAULT_MAX_PAIRS = 20
_MASK_CHUNK_BYTES = 64 * 1024 * 1024  # of float32 null mask held at once


def profile_dataset(
    dataset: Dataset,
    types: Mapping[str, ColumnType] | None = None,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    top_k: int = DEFAULT_TOP_K,
    max_bins: int = DEFAULT_MAX_BINS,
    max_pair_columns: int = DEFAULT_MAX_PAIR_COLUMNS,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> DatasetProfile:
    """The profile of a whole dataset: every column's, and the dataset's own.

    ``n_rows`` is the rows profiled, so a sample's rows rather than its source's.
    ``duplicate_rows`` counts rows equal to an earlier row, by a 64-bit row hash:
    over a million rows the chance of a false match is about 3e-8. It is ``None`` for
    a frame Polars cannot hash. Nulls follow ``ColumnCounts``: a Polars ``NaN`` is a
    value, not a null.

    ``types``, ``quantiles``, ``top_k`` and ``max_bins`` are those of
    ``profile_columns``. Missingness pairs the ``max_pair_columns`` columns with the
    most nulls and lists the ``max_pairs`` most similar pairs.

    Raises:
        ValueError: as ``profile_columns``, or ``max_pair_columns`` or ``max_pairs``
            is negative.
    """
    if max_pair_columns < 0 or max_pairs < 0:
        raise ValueError("max_pair_columns and max_pairs must not be negative")
    types = infer_column_types(dataset) if types is None else types
    columns = profile_columns(
        dataset, types, quantiles=quantiles, top_k=top_k, max_bins=max_bins
    )
    frame = dataset.frame
    return DatasetProfile(
        n_rows=frame.height,
        approx_memory_bytes=frame.estimated_size(),
        duplicate_rows=_duplicate_rows(frame),
        rows_with_any_null=_rows_with_any_null(frame),
        missingness=_missingness(frame, columns, max_pair_columns, max_pairs),
        columns=columns,
    )


def _duplicate_rows(frame: pl.DataFrame) -> int | None:
    if frame.width == 0:
        return None
    try:
        hashes = frame.hash_rows()
    except pl.exceptions.PolarsError:
        return None
    return frame.height - hashes.n_unique()


def _rows_with_any_null(frame: pl.DataFrame) -> int:
    if frame.width == 0 or frame.height == 0:
        return 0
    return int(frame.select(pl.any_horizontal(pl.all().is_null()).sum()).item())


def _missingness(
    frame: pl.DataFrame,
    columns: tuple[ColumnProfile, ...],
    max_columns: int,
    max_pairs: int,
) -> Missingness:
    with_nulls = [c for c in columns if c.counts.nulls > 0]
    # Most nulls first; the stable sort keeps column order among equals.
    chosen = sorted(with_nulls, key=lambda c: -c.counts.nulls)[:max_columns]
    chosen.sort(key=lambda c: c.position)
    pairs: tuple[NullPair, ...] = ()
    if len(chosen) >= 2 and max_pairs > 0:
        pairs = _top_pairs(frame, chosen, max_pairs)
    return Missingness(
        columns_with_nulls=len(with_nulls),
        columns_considered=len(chosen),
        pairs=pairs,
    )


def _top_pairs(
    frame: pl.DataFrame, chosen: list[ColumnProfile], max_pairs: int
) -> tuple[NullPair, ...]:
    names = [c.name for c in chosen]
    nulls = np.array([c.counts.nulls for c in chosen], dtype=np.int64)
    both = _shared_null_counts(frame.select(names))
    first, second = np.triu_indices(len(names), k=1)
    shared = both[first, second]
    present = shared > 0
    first, second, shared = first[present], second[present], shared[present]
    either = nulls[first] + nulls[second] - shared
    jaccard = shared / either
    # Most similar first, then most shared rows; lexsort is stable on column order.
    order = np.lexsort((-shared, -jaccard))[:max_pairs]
    return tuple(
        NullPair(
            a=names[first[i]],
            b=names[second[i]],
            both_null=int(shared[i]),
            jaccard=float(jaccard[i]),
        )
        for i in order
    )


def _shared_null_counts(frame: pl.DataFrame) -> np.ndarray:
    """Rows null in both columns, for every pair, from chunked ``maskᵀ @ mask``.

    float32 counts are exact up to 2**24 rows, which a chunk never reaches.
    """
    width = frame.width
    chunk = max(1, min(frame.height, _MASK_CHUNK_BYTES // (4 * width)))
    total = np.zeros((width, width), dtype=np.int64)
    for offset in range(0, frame.height, chunk):
        mask = (
            frame.slice(offset, chunk)
            .select(pl.all().is_null())
            .to_numpy()
            .astype(np.float32)
        )
        total += np.rint(mask.T @ mask).astype(np.int64)
    return total
