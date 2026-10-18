# dsbf/profile/histogram.py
"""Equal-width histograms of a numeric column, with the binning rule recorded.

The bin count comes from the Freedman–Diaconis rule, ``2 · IQR · n^(-1/3)`` for the
bin width, and from Sturges' rule, ``ceil(log2 n) + 1`` bins, when the interquartile
range is zero and FD has no width to offer. Either count is clipped to ``max_bins``,
because a heavy tail makes FD ask for more bins than a column has values.

Bins are ``[a, b)`` and the last is closed, so every finite value falls in exactly one
and the counts add up to the number of finite values.
"""

import math

import numpy as np
import polars as pl

from dsbf.model.profile import Histogram

DEFAULT_MAX_BINS = 100


def histogram(
    series: pl.Series,
    *,
    max_bins: int = DEFAULT_MAX_BINS,
    quartiles: tuple[float, float] | None = None,
) -> Histogram | None:
    """The histogram of ``series``' finite values, or None when it has no spread.

    ``quartiles`` are the 25th and 75th percentiles when the caller already has them;
    otherwise they are computed. None comes back for a column with no finite value,
    with one distinct finite value, or whose range is not a finite number.
    """
    values = _finite_values(series)
    if values.size == 0:
        return None
    low, high = float(values.min()), float(values.max())
    spread = high - low
    if not math.isfinite(spread) or spread <= 0:
        return None

    q1, q3 = quartiles if quartiles is not None else np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    if iqr > 0:
        rule = "fd"
        proposed = spread / (2 * iqr * values.size ** (-1 / 3))
    else:
        rule = "sturges"
        proposed = math.ceil(math.log2(values.size)) + 1
    bins = max(1, min(math.ceil(min(proposed, max_bins)), max_bins))

    if series.dtype.is_integer():
        edges = _integer_edges(low, spread, bins)
    else:
        edges = _float_edges(low, high, bins)
    where = np.searchsorted(edges, values, side="right") - 1
    counts = np.bincount(np.clip(where, 0, edges.size - 2), minlength=edges.size - 1)
    return Histogram(
        edges=tuple(edges.tolist()), counts=tuple(counts.tolist()), rule=rule
    )


def _finite_values(series: pl.Series) -> np.ndarray:
    as_float = series.cast(pl.Float64)
    return as_float.filter(as_float.is_finite()).to_numpy()


def _integer_edges(low: float, spread: float, bins: int) -> np.ndarray:
    # Whole-number widths, so no bin sits between two integers; the range covers
    # `spread + 1` integers and the edges run past the maximum.
    width = max(1, math.ceil((spread + 1) / bins))
    count = math.ceil((spread + 1) / width)
    return low + width * np.arange(count + 1, dtype=np.float64)


def _float_edges(low: float, high: float, bins: int) -> np.ndarray:
    # A range that is tiny next to its magnitude cannot hold many distinct edges.
    while True:
        edges = np.linspace(low, high, bins + 1)
        if bins == 1 or bool(np.all(np.diff(edges) > 0)):
            return edges
        bins //= 2
