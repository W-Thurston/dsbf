# dsbf/checks/pairs.py
"""What the pairwise checks share: missing values, codes, bands and intervals.

This module holds no check. ``associations``, ``contingency``, ``group_differences``
and ``collinearity`` import it, so they read a pair the same way and put its
strength on one scale.
"""

from typing import Literal

import numpy as np
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import norm

Band = Literal["negligible", "weak", "moderate", "strong"]
Metric = Literal["pearson_r", "spearman_r", "kendalls_tau", "eta", "cramers_v"]

# Lower edge of the weak, moderate and strong bands of |value|. The correlation
# family keeps the cut points of the task this replaces; η is banded at the η of the
# conventional η² cut points (0.01, 0.06, 0.14), so that it reads like a correlation.
BAND_EDGES: dict[Metric, tuple[float, float, float]] = {
    "pearson_r": (0.2, 0.4, 0.7),
    "spearman_r": (0.2, 0.4, 0.7),
    "kendalls_tau": (0.2, 0.4, 0.7),
    "eta": (0.1, 0.06**0.5, 0.14**0.5),
    "cramers_v": (0.1, 0.3, 0.5),
}
_BANDS: tuple[Band, ...] = ("negligible", "weak", "moderate", "strong")

# Variance of the Fisher-transformed statistic times (n - offset) (Fieller, Hartley and
# Pearson 1957): 1 for Pearson's r, 1.06 for Spearman's rho, 0.437 for Kendall's tau.
_FISHER: dict[str, tuple[float, int]] = {
    "pearson_r": (1.0, 3),
    "spearman_r": (1.06, 3),
    "kendalls_tau": (0.437, 4),
}

_BELOW_ONE = 1 - 1e-12


class Undefined(Exception):
    """A statistic that does not exist for these values; the reason is the message."""


def unit_column(values: np.ndarray) -> np.ndarray:
    """``values`` centred and scaled to length 1; correlation is a dot product.

    The column is divided by its largest magnitude first, so values near 1e308 do not
    overflow. Raises ``Undefined`` for a column with no spread or one that overflows.
    """
    with np.errstate(all="ignore"):
        biggest = float(np.max(np.abs(values)))
        if not np.isfinite(biggest):
            raise Undefined("overflow")
        if biggest == 0:
            raise Undefined("no_variation")
        scaled = values / biggest
        centred = scaled - scaled.mean()
        length = float(np.sqrt(np.sum(centred * centred)))
    if not np.isfinite(length):
        raise Undefined("overflow")
    if length == 0:
        raise Undefined("no_variation")
    return centred / length


def strength_band(value: float, metric: Metric) -> Band:
    """The band of ``abs(value)`` on the scale of its metric."""
    weak, moderate, strong = BAND_EDGES[metric]
    size = abs(value)
    return _BANDS[(size >= weak) + (size >= moderate) + (size >= strong)]


def band_rank(band: Band) -> int:
    return _BANDS.index(band)


def fisher_interval(
    value: float, n: int, metric: Metric, level: float
) -> tuple[float, float] | None:
    """Confidence interval for r, rho or tau by the Fisher z transform.

    None when ``n`` leaves no degrees of freedom, so a pair never gets an interval it
    cannot support.
    """
    variance, offset = _FISHER[metric]
    if n - offset < 1:
        return None
    z = np.arctanh(np.clip(value, -_BELOW_ONE, _BELOW_ONE))
    half = norm.ppf(0.5 + level / 2) * np.sqrt(variance / (n - offset))
    return float(np.tanh(z - half)), float(np.tanh(z + half))


def finite(values: np.ndarray) -> np.ndarray:
    """True where a numeric array holds a finite value (nulls arrive as NaN)."""
    return np.isfinite(values)


def level_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes of a categorical column, with -1 where it is missing.

    ``values`` is the column read with ``dropna=False``: objects holding ``None`` for
    text and booleans with nulls, floats with NaN for integer codes with nulls. Also
    returns the distinct levels, sorted, which the codes index.
    """
    codes = np.full(values.size, -1, dtype=np.int64)
    if values.dtype == object:
        present = np.fromiter(
            (v is not None and v == v for v in values), dtype=bool, count=values.size
        )
        keys = np.array([str(v) for v in values[present]]) if present.any() else None
    else:
        present = (
            ~np.isnan(values)
            if values.dtype.kind == "f"
            else np.ones(values.size, dtype=bool)
        )
        keys = values[present] if present.any() else None
    if keys is None:
        return codes, np.empty(0, dtype=object)
    levels, inverse = np.unique(keys, return_inverse=True)
    codes[present] = inverse
    return codes, levels


def correlation_ratio(
    x: np.ndarray, groups: np.ndarray, n_levels: int, *, min_n: int
) -> tuple[float, int]:
    """η of a numeric column on a grouping, and the rows it used.

    ``x`` holds finite values and ``groups`` their integer codes in ``[0, n_levels)``.
    A group of one row is fitted exactly, which inflates η, so such groups are left
    out. Raises ``Undefined`` for fewer than two groups of two or more rows
    (``too_few_groups``), fewer than ``min_n`` rows left (``too_few_rows``), no
    spread (``no_variation``) or an overflow (``overflow``).
    """
    counts = np.bincount(groups, minlength=n_levels)
    big = counts[groups] >= 2
    x, groups = x[big], groups[big]
    if np.count_nonzero(counts >= 2) < 2:
        raise Undefined("too_few_groups")
    n_used = int(x.size)
    if n_used < min_n:
        raise Undefined("too_few_rows")
    with np.errstate(all="ignore"):
        biggest = float(np.max(np.abs(x)))
        if not np.isfinite(biggest):
            raise Undefined("overflow")
        if biggest == 0:
            raise Undefined("no_variation")
        x = x / biggest
        x = x - x.mean()
        counts = np.bincount(groups, minlength=n_levels)
        sums = np.bincount(groups, weights=x, minlength=n_levels)
        present = counts > 0
        between = float(np.sum(sums[present] ** 2 / counts[present]))
        total = float(np.sum(x * x))
    if not np.isfinite(total):
        raise Undefined("overflow")
    if total == 0:
        raise Undefined("no_variation")
    return float(np.sqrt(np.clip(between / total, 0.0, 1.0))), n_used


def contingency_table(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts of the rows where both codes are present, without empty lines.

    Also returns the codes of ``first`` and ``second`` that the table's rows and
    columns stand for.
    """
    both = (first >= 0) & (second >= 0)
    a, b = first[both], second[both]
    if not a.size:
        empty = np.zeros(0, dtype=np.int64)
        return np.zeros((0, 0), dtype=np.int64), empty, empty
    width = int(b.max()) + 1
    table = np.bincount(a * width + b, minlength=(int(a.max()) + 1) * width)
    table = table.reshape(-1, width)
    rows = np.flatnonzero(table.sum(axis=1) > 0)
    cols = np.flatnonzero(table.sum(axis=0) > 0)
    return table[rows][:, cols], rows, cols


def cramers_v(tables: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chi-squared, plain V and bias-corrected V of each table (Bergsma 2013).

    ``tables`` has shape (..., rows, columns) and holds counts; the statistics have
    the shape of the leading dimensions. Rows and columns with no counts are not
    counted, which matters for a resampled table. V-bar is 0 where the correction
    leaves a table with fewer than two effective rows or columns.
    """
    tables = np.asarray(tables, dtype=np.float64)
    n = tables.sum(axis=(-2, -1))
    rows = tables.sum(axis=-1)
    cols = tables.sum(axis=-2)
    n_rows = (rows > 0).sum(axis=-1)
    n_cols = (cols > 0).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = rows[..., :, None] * cols[..., None, :] / n[..., None, None]
        terms = np.where(expected > 0, tables**2 / expected, 0.0)
        chi2 = terms.sum(axis=(-2, -1)) - n
        phi2 = np.where(n > 0, np.maximum(chi2, 0) / n, 0.0)
        plain = np.sqrt(
            np.where(
                np.minimum(n_rows, n_cols) > 1,
                phi2 / (np.minimum(n_rows, n_cols) - 1),
                0.0,
            )
        )
        shrink = np.where(n > 1, 1 / (n - 1), np.inf)
        phi2_tilde = np.maximum(0.0, phi2 - (n_rows - 1) * (n_cols - 1) * shrink)
        rows_tilde = n_rows - (n_rows - 1) ** 2 * shrink
        cols_tilde = n_cols - (n_cols - 1) ** 2 * shrink
        scale = np.minimum(rows_tilde, cols_tilde) - 1
        corrected = np.sqrt(np.where(scale > 0, phi2_tilde / scale, 0.0))
    return chi2, plain, np.minimum(corrected, 1.0)


def chi2_p_value(chi2: float, table: np.ndarray) -> float:
    """The upper-tail probability of the chi-squared statistic of a table."""
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    return float(chi2_distribution.sf(chi2, dof)) if dof > 0 else 1.0


Correction = Literal["fdr_bh", "bonferroni", "holm", "none"]


def adjust_p_values(p_values: np.ndarray, method: Correction) -> np.ndarray:
    """The p-values adjusted for the number of tests, in the order given, at most 1.

    ``fdr_bh`` is Benjamini-Hochberg (the false discovery rate), ``bonferroni`` and
    ``holm`` control the chance of any false positive, and ``none`` returns them as
    they are.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if method == "none" or n <= 1:
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * n, 1.0)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    if method == "holm":
        stepped = np.maximum.accumulate(ranked * (n - np.arange(n)))
    else:
        stepped = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(stepped, 1.0)
    return adjusted
