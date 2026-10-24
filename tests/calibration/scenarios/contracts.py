"""Data that breaks a stated expectation: bounds, a schema, a balance, a format."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, plant


@scenario("bounds_planted")
def bounds_planted(seed: int) -> Planted:
    """An age column with 2% of values over 120 and 1% below 0; a clean income."""
    rng = np.random.default_rng(seed)
    n = 2000
    age = rng.normal(45, 15, n).clip(1, 100)
    age[rng.choice(n, 40, replace=False)] = rng.uniform(121, 200, 40)
    age[rng.choice(n, 20, replace=False)] = rng.uniform(-30, -1, 20)
    frame = pd.DataFrame({"age": age, "income": rng.lognormal(10, 0.5, n)})
    return Planted(
        frame,
        plants=(plant("out_of_bounds", "age"),),
        params={
            "out_of_bounds": {
                "bounds": {"age": {"min": 0, "max": 120}, "income": {"min": 0}}
            }
        },
        checks=("out_of_bounds",),
    )


@scenario("bounds_clean")
def bounds_clean(seed: int) -> Planted:
    """Columns inside the bounds they are given."""
    rng = np.random.default_rng(seed)
    n = 2000
    frame = pd.DataFrame(
        {
            "age": rng.normal(45, 15, n).clip(1, 100),
            "income": rng.lognormal(10, 0.5, n),
        }
    )
    return Planted(
        frame,
        params={
            "out_of_bounds": {
                "bounds": {"age": {"min": 0, "max": 120}, "income": {"min": 0}}
            }
        },
        checks=("out_of_bounds",),
    )


@scenario("schema_planted")
def schema_planted(seed: int) -> Planted:
    """A required column is missing, a status holds a value outside its list."""
    rng = np.random.default_rng(seed)
    n = 1000
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "status": rng.choice(["open", "closed", "paused"], n, p=[0.5, 0.45, 0.05]),
            "score": rng.normal(0, 1, n),
        }
    )
    return Planted(
        frame,
        plants=(
            plant("schema", codes=["missing_column"]),
            plant("schema", "status"),
        ),
        params={
            "schema": {
                "columns": {
                    "customer_id": {"required": True},
                    "age": {"min": 0, "max": 120},
                    "status": {"allowed": ["open", "closed"]},
                    "score": {"type": "numeric"},
                }
            }
        },
        checks=("schema",),
    )


@scenario("imbalance_severe")
def imbalance_severe(seed: int) -> Planted:
    """A binary target with 3% positives."""
    rng = np.random.default_rng(seed)
    n = 2000
    frame = pd.DataFrame(
        {"target": (rng.random(n) < 0.03).astype(int), "x": rng.normal(0, 1, n)}
    )
    return Planted(
        frame,
        plants=(plant("class_imbalance", "target"),),
        target="target",
        checks=("class_imbalance",),
    )


@scenario("imbalance_none")
def imbalance_none(seed: int) -> Planted:
    """A binary target with 40% positives and a three-class one with 45/35/20."""
    rng = np.random.default_rng(seed)
    n = 2000
    frame = pd.DataFrame(
        {
            "target": (rng.random(n) < 0.4).astype(int),
            "grade": rng.choice(["a", "b", "c"], n, p=[0.45, 0.35, 0.2]),
        }
    )
    return Planted(frame, target="target", checks=("class_imbalance",))


@scenario("dates_unparsed")
def dates_unparsed(seed: int) -> Planted:
    """A date column with 8% of its values not dates, and a clean one."""
    rng = np.random.default_rng(seed)
    n = 1000
    stamps = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 500, n), "D")
    broken = stamps.strftime("%Y-%m-%d").to_numpy().astype(object)
    broken[rng.random(n) < 0.08] = "not a date"
    frame = pd.DataFrame({"opened": broken, "closed": stamps.strftime("%Y-%m-%d")})
    return Planted(
        frame,
        plants=(plant("datetime_consistency", "opened"),),
        checks=("datetime_consistency",),
    )


@scenario("sample_size_small")
def sample_size_small(seed: int) -> Planted:
    """Forty rows of twelve numeric features."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"x{i:02d}": rng.normal(0, 1, 40) for i in range(12)})
    return Planted(frame, plants=(plant("sample_size"),), checks=("sample_size",))


@scenario("sample_size_ample")
def sample_size_ample(seed: int) -> Planted:
    """Twenty thousand rows of twelve numeric features."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"x{i:02d}": rng.normal(0, 1, 20000) for i in range(12)})
    return Planted(frame, checks=("sample_size",))
