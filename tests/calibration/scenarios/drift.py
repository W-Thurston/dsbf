"""A dataset compared with a reference it has, or has not, moved away from."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, plant


def _population(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "level": rng.normal(100, 15, n),
            "spread": rng.normal(0, 1, n),
            "stable": rng.normal(5, 2, n),
            "skewed": rng.lognormal(0, 0.6, n),
            "segment": rng.choice(["a", "b", "c", "d"], n, p=[0.4, 0.3, 0.2, 0.1]),
            "channel": rng.choice(["web", "shop", "phone"], n, p=[0.5, 0.3, 0.2]),
            "amount": rng.normal(20, 4, n),
        }
    )


@scenario("drift_none", seeds=30)
def drift_none(seed: int) -> Planted:
    """Two independent samples of one population, 2 000 rows each."""
    rng = np.random.default_rng(seed)
    return Planted(
        _population(rng, 2000),
        reference=_population(rng, 2000),
        checks=("reference_drift",),
    )


@scenario("drift_none_small", seeds=30)
def drift_none_small(seed: int) -> Planted:
    """Two independent samples of one population, 150 rows each."""
    rng = np.random.default_rng(seed)
    return Planted(
        _population(rng, 150),
        reference=_population(rng, 150),
        checks=("reference_drift",),
    )


@scenario("drift_planted")
def drift_planted(seed: int) -> Planted:
    """A shifted mean, a wider spread, new category shares, and more nulls."""
    rng = np.random.default_rng(seed)
    n = 2000
    reference = _population(rng, n)
    current = _population(rng, n)
    current["level"] = current["level"] + 0.5 * 15
    current["spread"] = current["spread"] * 2
    current["segment"] = rng.choice(["a", "b", "c", "d"], n, p=[0.1, 0.2, 0.3, 0.4])
    amount = current["amount"].to_numpy().copy()
    amount[rng.random(n) < 0.3] = np.nan
    current["amount"] = amount
    return Planted(
        current,
        reference=reference,
        plants=(
            plant("reference_drift", "level"),
            plant("reference_drift", "spread"),
            plant("reference_drift", "segment"),
            plant("reference_drift", "amount", codes=["missingness_shift"]),
        ),
        checks=("reference_drift",),
    )


@scenario("drift_small_shift")
def drift_small_shift(seed: int) -> Planted:
    """A mean 0.1 deviations higher and a category share 3 points higher."""
    rng = np.random.default_rng(seed)
    n = 5000
    reference = _population(rng, n)
    current = _population(rng, n)
    current["level"] = current["level"] + 0.1 * 15
    current["segment"] = rng.choice(["a", "b", "c", "d"], n, p=[0.43, 0.3, 0.17, 0.1])
    return Planted(
        current,
        reference=reference,
        # Moves this small are real, but nobody needs to be told.
        allowed=(
            plant("reference_drift", "level"),
            plant("reference_drift", "segment"),
        ),
        checks=("reference_drift",),
    )


@scenario("drift_target")
def drift_target(seed: int) -> Planted:
    """The target's positive rate moves from 20% to 35%; the features do not."""
    rng = np.random.default_rng(seed)
    n = 2000
    reference = _population(rng, n)
    current = _population(rng, n)
    reference["churned"] = (rng.random(n) < 0.20).astype(int)
    current["churned"] = (rng.random(n) < 0.35).astype(int)
    return Planted(
        current,
        reference=reference,
        plants=(plant("reference_drift", "churned", codes=["target_drift"]),),
        target="churned",
        checks=("reference_drift",),
    )
