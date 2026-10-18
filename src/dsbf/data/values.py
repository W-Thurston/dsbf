# dsbf/data/values.py
"""NumPy arrays of a column's values, cached within a memory bound."""

import threading
from collections import OrderedDict

import numpy as np
import polars as pl

DEFAULT_CACHE_BYTES = 256 * 1024 * 1024  # 1M float64 values per column, 32 columns


class ValueCache:
    """A least-recently-used cache of arrays, bounded by their size in bytes.

    An array bigger than the whole bound is handed back but not kept, so the bound
    holds. The lock covers the cache only; callers convert outside it.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._entries: OrderedDict[tuple[str, bool], tuple[np.ndarray, int]] = (
            OrderedDict()
        )
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: tuple[str, bool]) -> np.ndarray | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: tuple[str, bool], array: np.ndarray, cost: int) -> None:
        if cost > self.max_bytes:
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            self._entries[key] = (array, cost)
            self._bytes += cost
            while self._bytes > self.max_bytes:
                _, (_, evicted_cost) = self._entries.popitem(last=False)
                self._bytes -= evicted_cost


def column_values(series: pl.Series, *, dropna: bool) -> tuple[np.ndarray, int]:
    """Return the column as a read-only array and its approximate size in bytes.

    ``dropna`` drops nulls and float NaN, which is what pandas calls missing, and
    keeps infinities. Without it, integers with nulls become float with NaN, and
    booleans and text with nulls become objects holding ``None``.
    """
    if dropna:
        series = series.drop_nulls()
        if series.dtype.is_float():
            series = series.drop_nans()
    if isinstance(series.dtype, pl.Struct):
        array = np.empty(len(series), dtype=object)
        array[:] = series.to_list()
    else:
        array = series.to_numpy()
    cost = array.nbytes
    if array.dtype == object:
        # `nbytes` counts the pointers, not the strings or lists they point to.
        cost += int(series.estimated_size())
    array.flags.writeable = False
    return array, cost
