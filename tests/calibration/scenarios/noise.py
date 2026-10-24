"""Frames with nothing wrong in them: every alarm on one is a false alarm."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, plant

# The uniform columns are the one thing in these frames that is not normal.
NOT_NORMAL = (plant("normality", "uniform_a"), plant("normality", "uniform_b"))


def independent_frame(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Independent, well-behaved columns of the kinds a clean table holds."""
    levels = [list("abcd"), ["north", "south", "east", "west", "centre"], list("xyz")]
    frame = {
        "normal_a": rng.normal(10, 2, n),
        "normal_b": rng.normal(-3, 0.5, n),
        "normal_c": rng.normal(100, 15, n),
        "uniform_a": rng.uniform(0, 1, n),
        "uniform_b": rng.uniform(-50, 50, n),
        "cat_a": rng.choice(levels[0], n),
        "cat_b": rng.choice(levels[1], n),
        "cat_c": rng.choice(levels[2], n, p=[0.5, 0.3, 0.2]),
        "flag": rng.random(n) < 0.4,
    }
    return pd.DataFrame(frame)


@scenario("noise_numeric")
def noise_numeric(seed: int) -> Planted:
    """3 000 rows of independent normal and uniform numbers and categoricals."""
    return Planted(
        independent_frame(np.random.default_rng(seed), 3000), plants=NOT_NORMAL
    )


@scenario("noise_small")
def noise_small(seed: int) -> Planted:
    """The same independent columns at 150 rows, where tests have little power."""
    return Planted(
        independent_frame(np.random.default_rng(seed), 150),
        plants=(*NOT_NORMAL, plant("sample_size")),
    )


@scenario("noise_large", seeds=5)
def noise_large(seed: int) -> Planted:
    """The independent columns at 30 000 rows, where small effects test as real."""
    return Planted(
        independent_frame(np.random.default_rng(seed), 30000), plants=NOT_NORMAL
    )
