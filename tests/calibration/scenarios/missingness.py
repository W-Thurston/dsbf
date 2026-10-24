"""Columns with missing values that are random, depend on others, or on themselves."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, plant

CHECKS = ("missingness_mechanism",)


def _predictors(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": rng.normal(45, 12, n),
            "tenure": rng.exponential(5, n),
            "score": rng.normal(0, 1, n),
            "region": rng.choice(["north", "south", "east", "west"], n),
            "plan": rng.choice(["basic", "plus"], n, p=[0.7, 0.3]),
        }
    )


def _null(rng: np.random.Generator, values: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    out[rng.random(len(values)) < p] = np.nan
    return out


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


@scenario("missing_at_random_noise")
def missing_at_random_noise(seed: int) -> Planted:
    """15% of one column is null, independent of five other columns."""
    rng = np.random.default_rng(seed)
    n = 2000
    frame = _predictors(rng, n)
    frame["income"] = _null(rng, rng.normal(50, 10, n), np.full(n, 0.15))
    return Planted(frame, checks=CHECKS)


@scenario("missing_completely_wide", seeds=10)
def missing_completely_wide(seed: int) -> Planted:
    """Four columns are 20% null at random among 30 numeric and 4 level predictors."""
    rng = np.random.default_rng(seed)
    n = 1500
    columns = {f"x{i:02d}": rng.normal(0, 1, n) for i in range(30)}
    for i in range(4):
        columns[f"c{i}"] = rng.choice(list("abcde"), n)
    for i in range(4):
        columns[f"holey_{i}"] = _null(rng, rng.normal(0, 1, n), np.full(n, 0.2))
    return Planted(pd.DataFrame(columns), checks=CHECKS)


@scenario("missing_at_random_planted")
def missing_at_random_planted(seed: int) -> Planted:
    """A column goes missing more often for older customers and in one region."""
    rng = np.random.default_rng(seed)
    n = 2000
    frame = _predictors(rng, n)
    p = _logistic(-1.5 + 0.08 * (frame["age"].to_numpy() - 45))
    p = np.where(frame["region"] == "south", np.minimum(1.0, p * 2.5), p)
    frame["income"] = _null(rng, rng.normal(50, 10, n), p)
    return Planted(
        frame,
        plants=(plant("missingness_mechanism", "income"),),
        checks=CHECKS,
    )


@scenario("missing_not_at_random_proxy")
def missing_not_at_random_proxy(seed: int) -> Planted:
    """Large incomes go missing, and spend, correlated with income, shows it."""
    rng = np.random.default_rng(seed)
    n = 2000
    frame = _predictors(rng, n)
    income = rng.normal(50, 10, n)
    frame["spend"] = 0.7 * (income - 50) / 10 + rng.normal(0, 0.7, n)
    frame["income"] = _null(rng, income, _logistic(2.5 * (income - 55) / 10 - 1.5))
    return Planted(
        frame,
        plants=(plant("missingness_mechanism", "income"),),
        checks=CHECKS,
    )


@scenario("missing_not_at_random_hidden")
def missing_not_at_random_hidden(seed: int) -> Planted:
    """Large incomes go missing and nothing observed is related to income."""
    rng = np.random.default_rng(seed)
    n = 2000
    frame = _predictors(rng, n)
    income = rng.normal(50, 10, n)
    frame["income"] = _null(rng, income, _logistic(2.5 * (income - 55) / 10 - 1.5))
    return Planted(frame, checks=CHECKS)
