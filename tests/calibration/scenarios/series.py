"""Series with a known shape against a datetime index: noise, AR(1), trend, cycle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, plant

SERIES_CHECKS = ("stationarity", "autocorrelation", "seasonal_decomposition")


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    shocks = rng.normal(0, 1, n)
    out = np.empty(n)
    out[0] = shocks[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


def _dates(n: int, freq: str = "D") -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=n, freq=freq)


@scenario("series_shapes")
def series_shapes(seed: int) -> Planted:
    """730 days of noise, AR(1), a walk, a ramp, a level shift and a weekly cycle."""
    rng = np.random.default_rng(seed)
    n = 730
    t = np.arange(n)
    frame = pd.DataFrame(
        {
            "date": _dates(n),
            "noise": rng.normal(0, 1, n),
            "ar1": _ar1(rng, n, 0.7),
            "walk": np.cumsum(rng.normal(0, 1, n)),
            "ramp": 0.05 * t + rng.normal(0, 1, n),
            "level_shift": np.where(t < n // 2, 0.0, 3.0) + rng.normal(0, 1, n),
            "weekly": 2 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, n),
        }
    )
    return Planted(
        frame,
        plants=(
            plant("autocorrelation", "ar1"),
            plant("autocorrelation", "walk"),
            plant("stationarity", "walk", codes=["non_stationary"]),
            plant("stationarity", "ramp", codes=["trend_stationary"]),
            plant("seasonal_decomposition", "weekly", codes=["seasonal"]),
            plant("seasonal_decomposition", "ramp", codes=["trend"]),
        ),
        allowed=(
            plant("autocorrelation", "ramp"),
            plant("autocorrelation", "level_shift"),
            plant("autocorrelation", "weekly"),
            plant("stationarity", "level_shift"),
            # A slowly wandering level is a trend to STL, whatever made it.
            plant("seasonal_decomposition", "ar1", codes=["trend"]),
            plant("seasonal_decomposition", "walk", codes=["trend"]),
            plant("seasonal_decomposition", "level_shift", codes=["trend"]),
        ),
        time_index="date",
        checks=SERIES_CHECKS,
    )


@scenario("series_white_noise", seeds=40)
def series_white_noise(seed: int) -> Planted:
    """Six columns of independent noise over 500 days."""
    rng = np.random.default_rng(seed)
    n = 500
    frame = pd.DataFrame({"date": _dates(n)})
    for i in range(6):
        frame[f"noise_{i}"] = rng.normal(0, 1, n)
    return Planted(frame, time_index="date", checks=SERIES_CHECKS)


@scenario("series_short_noise", seeds=40)
def series_short_noise(seed: int) -> Planted:
    """Six columns of independent noise over 48 months."""
    rng = np.random.default_rng(seed)
    n = 48
    frame = pd.DataFrame({"date": _dates(n, "MS")})
    for i in range(6):
        frame[f"noise_{i}"] = rng.normal(0, 1, n)
    return Planted(frame, time_index="date", checks=SERIES_CHECKS)


@scenario("series_gaps")
def series_gaps(seed: int) -> Planted:
    """Daily rows with a month missing and 10% of the other days dropped."""
    rng = np.random.default_rng(seed)
    days = _dates(730)
    keep = rng.random(len(days)) > 0.1
    keep[200:230] = False
    frame = pd.DataFrame({"date": days[keep], "value": rng.normal(0, 1, keep.sum())})
    return Planted(
        frame,
        plants=(plant("temporal_gaps"),),
        time_index="date",
        checks=("temporal_gaps",),
    )


@scenario("series_regular")
def series_regular(seed: int) -> Planted:
    """Daily rows, none missing."""
    rng = np.random.default_rng(seed)
    n = 730
    frame = pd.DataFrame({"date": _dates(n), "value": rng.normal(0, 1, n)})
    return Planted(frame, time_index="date", checks=("temporal_gaps",))
