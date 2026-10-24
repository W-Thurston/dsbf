"""Columns for which one preparation step is clearly right, or clearly wrong."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from dsbf.model import Finding
from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, plant


def methods(finding: Finding) -> set[str]:
    """The methods of the actions a finding advises."""
    if finding.guidance is None:
        return set()
    return {a.method for a in finding.guidance.actions if a.method}


def advises(*wanted: str) -> Callable[[Finding], bool]:
    """Whether a finding advises one of ``wanted``."""
    return lambda finding: bool(methods(finding) & set(wanted))


def avoids(*unwanted: str) -> Callable[[Finding], bool]:
    """Whether a finding advises none of ``unwanted``."""
    return lambda finding: not methods(finding) & set(unwanted)


@scenario("transform_planted")
def transform_planted(seed: int) -> Planted:
    """Skewed columns that a power transform fixes, and columns it must leave alone."""
    rng = np.random.default_rng(seed)
    n = 3000
    frame = pd.DataFrame(
        {
            "lognormal": rng.lognormal(1, 1.0, n),
            "exponential": rng.exponential(3, n),
            "left_skewed": 100 - rng.exponential(5, n),
            "signed_skewed": rng.lognormal(0, 1, n) - 2,
            "normal": rng.normal(10, 2, n),
            "uniform": rng.uniform(0, 1, n),
            "mild_skew": rng.gamma(16, 1, n),
        }
    )
    suggest = ("transform_suggested",)
    return Planted(
        frame,
        plants=tuple(
            plant("transformations", c, codes=suggest)
            for c in ("lognormal", "exponential", "left_skewed", "signed_skewed")
        ),
        # A skew near 0.5 is below the point where anyone needs to transform.
        allowed=(plant("transformations", "mild_skew"),),
        checks=("transformations",),
    )


@scenario("binning_planted")
def binning_planted(seed: int) -> Planted:
    """Columns that suit equal-width bins and columns that need bins of equal counts."""
    rng = np.random.default_rng(seed)
    n = 3000
    frame = pd.DataFrame(
        {
            "normal": rng.normal(0, 1, n),
            "uniform": rng.uniform(0, 100, n),
            "lognormal": rng.lognormal(0, 1, n),
            "exponential": rng.exponential(1, n),
            "long_tail": np.where(rng.random(n) < 0.002, 500.0, rng.normal(0, 1, n)),
        }
    )
    return Planted(
        frame,
        plants=(
            plant("numeric_binning", "normal", where=advises("equal_width")),
            plant("numeric_binning", "uniform", where=advises("equal_width")),
            plant("numeric_binning", "lognormal", where=advises("quantile")),
            plant("numeric_binning", "exponential", where=advises("quantile")),
            plant("numeric_binning", "long_tail", where=advises("quantile")),
        ),
        checks=("numeric_binning",),
    )


@scenario("encoding_planted")
def encoding_planted(seed: int) -> Planted:
    """A few levels, with plenty of rows each, and many levels, with few rows each."""
    rng = np.random.default_rng(seed)
    n = 3000
    frame = pd.DataFrame(
        {
            "target": rng.integers(0, 2, n),
            "few_levels": rng.choice(["red", "green", "blue", "grey"], n),
            "many_levels": rng.choice([f"c{i:03d}" for i in range(150)], n),
            "very_many_levels": rng.choice([f"v{i:04d}" for i in range(900)], n),
        }
    )
    return Planted(
        frame,
        plants=(
            plant("categorical_encoding", "few_levels", where=advises("one_hot")),
            plant(
                "categorical_encoding", "many_levels", where=avoids("one_hot", "binary")
            ),
            plant(
                "categorical_encoding",
                "very_many_levels",
                where=avoids("one_hot", "binary"),
            ),
        ),
        allowed=(plant("categorical_encoding", "target"),),
        target="target",
        checks=("categorical_encoding",),
    )


@scenario("imputation_planted")
def imputation_planted(seed: int) -> Planted:
    """Nulls in a symmetric column, a skewed one and a categorical one."""
    rng = np.random.default_rng(seed)
    n = 2000

    def holes(values: np.ndarray, share: float = 0.1) -> np.ndarray:
        out = values.astype(object)
        out[rng.random(n) < share] = None
        return out

    frame = pd.DataFrame(
        {
            "symmetric": holes(rng.normal(50, 5, n)),
            "skewed": holes(rng.lognormal(0, 1.2, n)),
            "category": holes(rng.choice(["a", "b", "c"], n, p=[0.6, 0.3, 0.1])),
        }
    )
    return Planted(
        frame,
        plants=(
            plant("imputation", "symmetric", where=advises("mean", "median")),
            plant("imputation", "skewed", where=advises("median")),
            plant("imputation", "category", where=advises("mode", "constant_missing")),
        ),
        checks=("imputation",),
    )


def _spiked(rng: np.random.Generator, n: int, share: float) -> np.ndarray:
    """Zero for all but ``share`` of the rows, which are normal around 10."""
    values = np.zeros(n)
    nonzero = rng.random(n) < share
    values[nonzero] = rng.normal(10, 3, int(nonzero.sum()))
    return values


def _transform_spike(seed: int, n: int) -> Planted:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "spike_5pct": _spiked(rng, n, 0.05),
            "spike_10pct": _spiked(rng, n, 0.10),
            "heavy_lognormal": rng.lognormal(0, 2, n),
        }
    )
    return Planted(
        frame,
        plants=(
            # A point mass cannot be spread by any power, whatever the row count.
            plant("transformations", "spike_5pct", codes=["no_effective_transform"]),
            plant("transformations", "spike_10pct", codes=["no_effective_transform"]),
            plant("transformations", "heavy_lognormal", codes=["transform_suggested"]),
        ),
        checks=("transformations",),
    )


@scenario("transform_spike_5000")
def transform_spike_5000(seed: int) -> Planted:
    """Columns that are mostly zero, at 5 000 rows, beside one that a log fixes."""
    return _transform_spike(seed, 5000)


@scenario("transform_spike_20000")
def transform_spike_20000(seed: int) -> Planted:
    """The same columns at 20 000 rows, where the fitted power moves."""
    return _transform_spike(seed, 20000)
