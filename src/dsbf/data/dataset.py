# dsbf/data/dataset.py
"""The loaded data every later stage works from."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import polars as pl

from dsbf.data.errors import SamplingError
from dsbf.data.fingerprint import fingerprint_frame
from dsbf.data.sampling import Strategy, check_request, sample_frame
from dsbf.data.values import DEFAULT_CACHE_BYTES, ValueCache, column_values
from dsbf.model.evidence import SamplingInfo
from dsbf.model.manifest import SourceRef


class _State:
    """What a Dataset works out once and keeps, so the dataclass itself stays frozen."""

    def __init__(self, cache_bytes: int) -> None:
        self.fingerprint: str | None = None
        self.values = ValueCache(cache_bytes)


@dataclass(frozen=True, slots=True)
class Dataset:
    """A Polars-backed table and where it came from.

    Treat ``frame`` as read-only. The fingerprint, a sample and the value cache are
    all taken from it when first asked, so an edit afterwards would leave them
    describing the old data. ``frame`` is not copied from a pandas or Arrow input
    where the memory can be shared: with pandas 2.x, editing the pandas frame in
    place after ``load`` changes ``frame`` too, so pass ``df.copy()`` to ``load`` if
    that can happen. pandas 3's copy-on-write makes it impossible.
    """

    frame: pl.DataFrame
    name: str
    source: SourceRef
    coerced: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}),
        metadata={"doc": "Columns that had to change type to load, with the reason."},
    )
    sampling: SamplingInfo | None = field(
        default=None,
        metadata={"doc": "How ``frame`` was drawn from the source; None if it is all."},
    )
    source_fingerprint: str | None = field(
        default=None,
        metadata={"doc": "For a sample, the fingerprint of the full data it is from."},
    )
    value_cache_bytes: int = field(
        default=DEFAULT_CACHE_BYTES,
        metadata={"doc": "Memory bound of the cache behind ``values``, in bytes."},
    )
    _state: _State = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (self.sampling is None) != (self.source_fingerprint is None):
            raise ValueError("a sample names the full data's fingerprint, and only it")
        if self.value_cache_bytes < 0:
            raise ValueError("value_cache_bytes cannot be negative")
        object.__setattr__(self, "_state", _State(self.value_cache_bytes))

    @property
    def n_rows(self) -> int:
        return self.frame.height

    @property
    def source_rows(self) -> int:
        """Rows before any sampling."""
        return self.sampling.original_rows if self.sampling else self.n_rows

    @property
    def n_columns(self) -> int:
        return self.frame.width

    @property
    def columns(self) -> list[str]:
        return self.frame.columns

    @property
    def schema(self) -> pl.Schema:
        return self.frame.schema

    @property
    def fingerprint(self) -> str:
        """A content hash of the full data: equal data, equal fingerprint.

        It covers column names, dtypes, row order and every value, so changing one
        cell changes it. A sample has its source's fingerprint, and ``sampling``
        says which rows of it these are.
        """
        if self.source_fingerprint is not None:
            return self.source_fingerprint
        state = self._state
        if state.fingerprint is None:
            state.fingerprint = fingerprint_frame(self.frame)
        return state.fingerprint

    def sample(
        self,
        rows: int,
        *,
        strategy: Strategy = "head",
        seed: int | None = None,
        stratify_by: Sequence[str] = (),
    ) -> "Dataset":
        """Return a dataset of ``rows`` rows with a record of how they were chosen.

        ``head`` takes the first rows. ``random`` takes a seeded random subset
        (seed 0 unless given). ``stratified`` takes the same number of rows with
        the groups of ``stratify_by`` in proportion (null is a group of its own),
        chosen at random with the seed; a group too small for a row of its share is
        left out. Row order is kept. A strategy or column that cannot be honoured
        raises :class:`SamplingError`; none falls back to ``head``.

        When ``rows`` is at least the table's, the dataset is returned as it is and
        ``sampling`` stays ``None``.

        Raises:
            SamplingError: the request is invalid, or this dataset is already a sample.
        """
        check_request(self.frame, rows, strategy, seed, stratify_by)
        if self.sampling is not None:
            raise SamplingError(
                "This dataset is already a sample; sample the full dataset instead"
            )
        if rows >= self.n_rows:
            return self
        frame, info = sample_frame(
            self.frame, rows, strategy=strategy, seed=seed, stratify_by=stratify_by
        )
        return Dataset(
            frame=frame,
            name=self.name,
            source=self.source,
            coerced=self.coerced,
            sampling=info,
            source_fingerprint=self.fingerprint,
            value_cache_bytes=self.value_cache_bytes,
        )

    def values(self, column: str, dropna: bool = True) -> np.ndarray:
        """The column as a read-only NumPy array, cached.

        ``dropna`` drops nulls and float NaN and keeps infinities. With
        ``dropna=False`` an integer column with nulls comes back as float with NaN,
        and boolean and text columns with nulls as objects holding ``None``. The
        same array is returned until the cache, bounded by ``value_cache_bytes``,
        evicts it; an array bigger than the bound is never kept.
        """
        if column not in self.frame.columns:
            raise KeyError(f"No column named {column!r} ({self.n_columns} columns)")
        cache = self._state.values
        key = (column, dropna)
        array = cache.get(key)
        if array is None:
            array, cost = column_values(self.frame.get_column(column), dropna=dropna)
            cache.put(key, array, cost)
        return array
