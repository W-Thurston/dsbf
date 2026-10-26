"""Columns with a planted shape: outliers, skew, heavy tails, two modes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dsbf.model import Severity
from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, Probe, plant

SHAPE_CHECKS = ("outliers", "normality", "skewness", "kurtosis", "bimodality")


def _with_spikes(
    rng: np.random.Generator, n: int, count: int, size: float
) -> np.ndarray:
    """A standard normal column with ``count`` values ``size`` deviations out."""
    values = rng.normal(0, 1, n)
    at = rng.choice(n, count, replace=False)
    values[at] = size * rng.choice([-1, 1], count)
    return values


@scenario("outliers_planted")
def outliers_planted(seed: int) -> Planted:
    """1% of one normal column sits ten deviations out; two normal columns are clean."""
    rng = np.random.default_rng(seed)
    n = 3000
    frame = pd.DataFrame(
        {
            "spiked": _with_spikes(rng, n, 30, 10),
            "clean_a": rng.normal(50, 5, n),
            "clean_b": rng.normal(0, 1, n),
        }
    )
    return Planted(
        frame,
        plants=(
            plant("outliers", "spiked"),
            plant("normality", "spiked"),
            plant("kurtosis", "spiked", codes=["heavy_tails"]),
        ),
        allowed=(plant("skewness", "spiked"),),
        checks=("outliers", "normality", "kurtosis", "skewness"),
    )


@scenario("outliers_rare")
def outliers_rare(seed: int) -> Planted:
    """Three values of one normal column sit seven deviations out in 3 000 rows."""
    rng = np.random.default_rng(seed)
    n = 3000
    frame = pd.DataFrame(
        {
            "spiked": _with_spikes(rng, n, 3, 7),
            "clean_a": rng.normal(50, 5, n),
            "clean_b": rng.normal(0, 1, n),
        }
    )
    return Planted(
        frame,
        plants=(plant("outliers", "spiked"),),
        allowed=(plant("normality", "spiked"), plant("kurtosis", "spiked")),
        checks=("outliers", "normality", "kurtosis"),
    )


@scenario("joint_outliers")
def joint_outliers(seed: int) -> Planted:
    """Two columns correlated at 0.9; 1% of rows break that inside the ranges."""
    rng = np.random.default_rng(seed)
    n = 3000
    x = rng.normal(0, 1, n)
    y = 0.9 * x + np.sqrt(1 - 0.81) * rng.normal(0, 1, n)
    broken = rng.choice(n, 30, replace=False)
    sign = rng.choice([-1, 1], 30)
    x[broken] = 1.8 * sign
    y[broken] = -1.8 * sign
    frame = pd.DataFrame({"x": x, "y": y, "z": rng.normal(0, 1, n)})
    return Planted(
        frame,
        plants=(plant("multivariate_outliers"),),
        allowed=(plant("outliers"),),
        checks=("multivariate_outliers", "outliers"),
    )


def _far_share(result, planted: Planted) -> float | None:
    if result.output is None:
        return None
    return result.output["far_rows"] / max(result.output["rows_used"], 1)


@scenario("joint_skewed_nulls", seeds=40)
def joint_skewed_nulls(seed: int) -> Planted:
    """Correlated lognormal and cubed-normal columns with nothing planted.

    Skew alone puts a quarter of the rows beyond a chi-squared limit that assumes a
    normal cloud, so the far-row share must stay near the stated rate.
    """
    rng = np.random.default_rng(seed)
    n = 1000
    correlation = np.full((4, 4), 0.5) + 0.5 * np.eye(4)
    z = rng.standard_normal((n, 4)) @ np.linalg.cholesky(correlation).T
    frame = pd.DataFrame(
        {
            "lognormal_a": np.exp(z[:, 0]),
            "lognormal_b": np.exp(z[:, 1]),
            "cubed_a": z[:, 2] ** 3,
            "cubed_b": z[:, 3] ** 3,
        }
    )
    return Planted(
        frame,
        checks=("multivariate_outliers",),
        probes=(Probe("multivariate_outliers", "far_share", _far_share, False),),
    )


@scenario("shapes_planted")
def shapes_planted(seed: int) -> Planted:
    """Seven columns of 3 000 values, each of a named shape, one of them normal."""
    rng = np.random.default_rng(seed)
    n = 3000
    modes = rng.random(n) < 0.5
    weak = rng.random(n) < 0.5
    frame = pd.DataFrame(
        {
            "normal": rng.normal(0, 1, n),
            "lognormal": rng.lognormal(0, 1, n),
            "exponential": rng.exponential(1, n),
            "uniform": rng.uniform(0, 1, n),
            "heavy_tailed": rng.standard_t(3, n),
            "bimodal": np.where(modes, rng.normal(-3, 1, n), rng.normal(3, 1, n)),
            "weakly_bimodal": np.where(weak, rng.normal(-1, 1, n), rng.normal(1, 1, n)),
        }
    )
    return Planted(
        frame,
        plants=(
            plant("skewness", "lognormal", codes=["skewed"]),
            plant("skewness", "exponential", codes=["skewed"]),
            plant("kurtosis", "heavy_tailed", codes=["heavy_tails"]),
            plant("bimodality", "bimodal"),
            *(
                plant("normality", c)
                for c in (
                    "lognormal",
                    "exponential",
                    "uniform",
                    "heavy_tailed",
                    "bimodal",
                )
            ),
        ),
        allowed=(
            # Two normals 2 deviations apart are barely bimodal and barely non-normal.
            plant("bimodality", "weakly_bimodal"),
            plant("normality", "weakly_bimodal"),
            plant("kurtosis", "lognormal"),
            plant("kurtosis", "exponential"),
            plant("outliers", "lognormal"),
            plant("outliers", "exponential"),
            plant("outliers", "heavy_tailed"),
            plant("skewness", "heavy_tailed"),
        ),
        checks=SHAPE_CHECKS,
    )


@scenario("bimodal_few_values", seeds=40)
def bimodal_few_values(seed: int) -> Planted:
    """200 values: two peaks 4 spreads apart, and a uniform and a normal column."""
    rng = np.random.default_rng(seed)
    n = 200
    frame = pd.DataFrame(
        {
            "two_peaks": np.where(
                rng.random(n) < 0.5, rng.normal(0, 1, n), rng.normal(4, 1, n)
            ),
            "uniform": rng.uniform(0, 1, n),
            "normal": rng.normal(0, 1, n),
        }
    )
    return Planted(
        frame, plants=(plant("bimodality", "two_peaks"),), checks=("bimodality",)
    )


@scenario("normality_large", seeds=20)
def normality_large(seed: int) -> Planted:
    """8 000 values: four normal columns, a lognormal one and a near-normal t(30)."""
    rng = np.random.default_rng(seed)
    n = 8000
    frame = pd.DataFrame(
        {
            **{f"normal_{i}": rng.normal(0, 1, n) for i in range(4)},
            "lognormal": rng.lognormal(0, 0.5, n),
            "near_normal": rng.standard_t(30, n),
        }
    )
    return Planted(
        frame,
        plants=(plant("normality", "lognormal"),),
        allowed=(
            plant("normality", "near_normal"),
            plant("skewness", "lognormal"),
            plant("kurtosis", "lognormal"),
        ),
        checks=("normality", "skewness", "kurtosis"),
    )


@scenario("mean_median_gap")
def mean_median_gap(seed: int) -> Planted:
    """Columns whose mean sits far from their median, without a tail to show it."""
    rng = np.random.default_rng(seed)
    n = 3000
    frame = pd.DataFrame(
        {
            "normal": rng.normal(10, 2, n),
            "zero_inflated": np.where(rng.random(n) < 0.55, 0.0, rng.exponential(5, n)),
            "two_groups": np.where(
                rng.random(n) < 0.7, rng.normal(0, 1, n), rng.normal(6, 1, n)
            ),
            "exponential": rng.exponential(1, n),
            "uniform": rng.uniform(0, 1, n),
        }
    )
    return Planted(
        frame,
        plants=(
            plant(
                "skewness",
                "zero_inflated",
                codes=["mean_median_gap"],
                min_severity=Severity.WARN,
            ),
            plant("skewness", "two_groups", codes=["mean_median_gap"]),
        ),
        # A tail pulls the mean off the median by a third of a deviation at most.
        allowed=(
            plant("skewness", "zero_inflated", codes=["skewed"]),
            plant("skewness", "two_groups", codes=["skewed"]),
            plant("skewness", "exponential"),
        ),
        checks=("skewness",),
    )
