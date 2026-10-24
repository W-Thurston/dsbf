"""Related, redundant and independent pairs of columns, and group differences."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, plant

PAIR_CHECKS = ("associations", "contingency", "group_differences", "collinearity")


def _correlated(rng: np.random.Generator, x: np.ndarray, r: float) -> np.ndarray:
    """A column correlated ``r`` with ``x`` (a standard normal)."""
    return r * x + np.sqrt(1 - r**2) * rng.normal(0, 1, len(x))


def _dependent_levels(
    rng: np.random.Generator, first: np.ndarray, purity: float, levels: int
) -> np.ndarray:
    """Levels that repeat ``first`` a share ``purity`` of the time, else are random."""
    pool = np.array([f"l{i}" for i in range(levels)])
    follow = pool[first]
    return np.where(
        rng.random(len(first)) < purity, follow, rng.choice(pool, len(first))
    )


@scenario("pairs_numeric")
def pairs_numeric(seed: int) -> Planted:
    """Pairs correlated at 0.9, 0.8 and 0.2 beside two independent columns."""
    rng = np.random.default_rng(seed)
    n = 2000
    base = rng.normal(0, 1, n)
    other = rng.normal(0, 1, n)
    weak_base = rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            "base": base,
            "strong": _correlated(rng, base, 0.9),
            "other": other,
            "moderate": _correlated(rng, other, 0.8),
            "weak_base": weak_base,
            "weak": _correlated(rng, weak_base, 0.2),
            "free_a": rng.normal(0, 1, n),
            "free_b": rng.normal(0, 1, n),
        }
    )
    return Planted(
        frame,
        plants=(
            plant("associations", "base", "strong"),
            plant("associations", "other", "moderate"),
        ),
        allowed=(plant("associations", "weak_base", "weak"),),
        checks=("associations",),
    )


@scenario("pairs_independent", seeds=30)
def pairs_independent(seed: int) -> Planted:
    """Twelve independent numeric columns, so 66 pairs and no relation."""
    rng = np.random.default_rng(seed)
    n = 1500
    frame = pd.DataFrame({f"x{i:02d}": rng.normal(0, 1, n) for i in range(12)})
    return Planted(frame, checks=("associations", "collinearity"))


@scenario("pairs_categorical")
def pairs_categorical(seed: int) -> Planted:
    """Two level columns that agree 60% of the time, a weak pair, free columns."""
    rng = np.random.default_rng(seed)
    n = 2000
    codes = rng.integers(0, 4, n)
    weak = rng.integers(0, 4, n)
    frame = pd.DataFrame(
        {
            "driver": np.array([f"l{i}" for i in codes]),
            "follower": _dependent_levels(rng, codes, 0.6, 4),
            "weak_driver": np.array([f"l{i}" for i in weak]),
            "weak_follower": _dependent_levels(rng, weak, 0.12, 4),
            "free_a": rng.choice(list("abcde"), n),
            "free_b": rng.choice(list("xyz"), n),
        }
    )
    return Planted(
        frame,
        plants=(plant("contingency", "driver", "follower"),),
        allowed=(plant("contingency", "weak_driver", "weak_follower"),),
        checks=("contingency", "associations"),
    )


@scenario("groups_shifted")
def groups_shifted(seed: int) -> Planted:
    """A value that is a full deviation higher in one of four groups, and a free one."""
    rng = np.random.default_rng(seed)
    n = 2000
    group = rng.choice(list("abcd"), n)
    shifted = rng.normal(0, 1, n) + np.where(group == "a", 1.0, 0.0)
    tiny = rng.normal(0, 1, n) + np.where(group == "b", 0.05, 0.0)
    frame = pd.DataFrame(
        {
            "group": group,
            "shifted": shifted,
            "tiny_shift": tiny,
            "free": rng.normal(0, 1, n),
        }
    )
    return Planted(
        frame,
        plants=(plant("group_differences", "shifted", "group"),),
        allowed=(plant("group_differences", "tiny_shift", "group"),),
        checks=("group_differences",),
    )


@scenario("groups_unequal", seeds=40)
def groups_unequal(seed: int) -> Planted:
    """Equal means in groups of 600, 300, 100 and 30, the small ones the most spread."""
    rng = np.random.default_rng(seed)
    sizes = {
        "big": (600, 1.0),
        "mid": (300, 1.5),
        "small": (100, 3.0),
        "tiny": (30, 6.0),
    }
    group = np.concatenate([[name] * size for name, (size, _) in sizes.items()])
    value = np.concatenate([rng.normal(0, sd, size) for size, sd in sizes.values()])
    frame = pd.DataFrame({"group": group, "value": value})
    return Planted(frame, checks=("group_differences",))


@scenario("collinear_planted")
def collinear_planted(seed: int) -> Planted:
    """A column that is the sum of two others, a near-copy, and four free columns."""
    rng = np.random.default_rng(seed)
    n = 2000
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            "a": a,
            "b": b,
            "a_plus_b": a + b + rng.normal(0, 0.05, n),
            "c": rng.normal(0, 1, n),
            "c_copy": 3 * rng.normal(0, 1, n),
            "d": rng.normal(0, 1, n),
        }
    )
    frame["c_copy"] = 3 * frame["c"] + rng.normal(0, 0.1, n)
    return Planted(
        frame,
        plants=tuple(
            plant("collinearity", c) for c in ("a", "b", "a_plus_b", "c", "c_copy")
        ),
        checks=("collinearity",),
    )
