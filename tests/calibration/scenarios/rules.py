"""Columns that break a plain rule: nulls, constants, zeros, identifiers, bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, plant

RULE_CHECKS = (
    "null_rate",
    "constant_columns",
    "near_zero_variance",
    "zeros",
    "dominant_value",
    "high_cardinality",
    "id_columns",
    "duplicate_columns",
    "text_encoding",
)


def _with_nulls(rng: np.random.Generator, values: pd.Series, share: float) -> pd.Series:
    out = values.astype("object")
    out[rng.random(len(values)) < share] = None
    return out


@scenario("rules_planted")
def rules_planted(seed: int) -> Planted:
    """Each column breaks one rule at a clear level; three columns are fine."""
    rng = np.random.default_rng(seed)
    n = 1000
    base = rng.normal(0, 1, n)
    zeros = np.where(rng.random(n) < 0.55, 0.0, rng.normal(5, 1, n))
    some_zeros = np.where(rng.random(n) < 0.08, 0.0, rng.normal(5, 1, n))
    dominant = rng.choice(["a", "b"], n, p=[0.96, 0.04])
    damaged = np.array([f"caf� {i % 7}" for i in range(n)], dtype=object)
    damaged[rng.random(n) >= 0.4] = "ok"
    frame = pd.DataFrame(
        {
            "fine_numeric": base,
            "fine_category": rng.choice(list("abcde"), n),
            "few_nulls": _with_nulls(rng, pd.Series(rng.normal(0, 1, n)), 0.03),
            "many_nulls": _with_nulls(rng, pd.Series(rng.normal(0, 1, n)), 0.3),
            "mostly_null": _with_nulls(rng, pd.Series(rng.normal(0, 1, n)), 0.85),
            "constant": np.full(n, 7),
            "frozen": 3.0 + rng.normal(0, 1e-6, n),
            "mostly_zero": zeros,
            "some_zeros": some_zeros,
            "dominant": dominant,
            "high_cardinality": rng.choice([f"level-{i}" for i in range(120)], n),
            "row_id": [f"ID-{i:05d}" for i in range(n)],
            "copy_a": base * 2 + 1,
            "damaged": damaged,
        }
    )
    frame["copy_b"] = frame["copy_a"]
    return Planted(
        frame,
        plants=(
            plant("null_rate", "many_nulls"),
            plant("null_rate", "mostly_null"),
            plant("constant_columns"),
            plant("near_zero_variance", "frozen"),
            plant("zeros", "mostly_zero"),
            plant("dominant_value", "dominant"),
            plant("high_cardinality", "high_cardinality"),
            plant("id_columns", "row_id"),
            plant("duplicate_columns"),
            plant("text_encoding", "damaged"),
        ),
        allowed=(plant("near_zero_variance", "constant"),),
        checks=RULE_CHECKS,
    )
