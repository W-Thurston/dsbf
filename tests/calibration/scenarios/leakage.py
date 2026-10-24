"""Features that carry the target, and informative features that must not be flagged."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, plant

LEAK_CHECKS = ("target_leakage", "mutual_information")


def _flip(rng: np.random.Generator, labels: np.ndarray, share: float) -> np.ndarray:
    """``labels`` of a two-class target with ``share`` of them swapped."""
    swapped = 1 - labels
    return np.where(rng.random(len(labels)) < share, swapped, labels)


def _signal(
    rng: np.random.Generator, target: np.ndarray, strength: float
) -> np.ndarray:
    """A numeric feature ``strength`` deviations higher when the target is positive."""
    return rng.normal(0, 1, len(target)) + strength * target


def _target_frame(rng: np.random.Generator, n: int) -> tuple[pd.DataFrame, np.ndarray]:
    target = (rng.random(n) < 0.4).astype(int)
    frame = pd.DataFrame(
        {
            "target": target,
            "signal": _signal(rng, target, 1.0),
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(0, 1, n),
            "group": rng.choice(list("abcd"), n),
        }
    )
    return frame, target


@scenario("leakage_exact")
def leakage_exact(seed: int) -> Planted:
    """A target, a copy, a recode, a scaled copy and one honest strong feature."""
    rng = np.random.default_rng(seed)
    frame, target = _target_frame(rng, 2000)
    frame["leak_copy"] = target
    frame["leak_recode"] = np.where(target == 1, "approved", "declined")
    frame["leak_scaled"] = target * 40.0 + 12
    frame["leak_noisy_scale"] = target * 40.0 + rng.normal(0, 0.5, len(target))
    return Planted(
        frame,
        plants=tuple(
            plant("target_leakage", c)
            for c in ("leak_copy", "leak_recode", "leak_scaled", "leak_noisy_scale")
        ),
        allowed=tuple(
            plant("mutual_information", c)
            for c in ("leak_copy", "leak_recode", "leak_scaled", "leak_noisy_scale")
        )
        + (plant("mutual_information", "signal"),),
        target="target",
        checks=LEAK_CHECKS,
    )


@scenario("leakage_flipped")
def leakage_flipped(seed: int) -> Planted:
    """Recodes of the target with 1%, 2%, 3%, 5% and 10% of the labels flipped."""
    rng = np.random.default_rng(seed)
    frame, target = _target_frame(rng, 2000)
    for share in (1, 2, 3, 5, 10):
        frame[f"flip_{share}pct"] = _flip(rng, target, share / 100)
    return Planted(
        frame,
        plants=(
            plant("target_leakage", "flip_1pct"),
            plant("target_leakage", "flip_2pct"),
        ),
        # A feature that is right 95% to 97% of the time is leakage or a very good
        # model; either way it is not a false alarm.
        allowed=(
            plant("target_leakage", "flip_3pct"),
            plant("target_leakage", "flip_5pct"),
        ),
        target="target",
        checks=("target_leakage",),
    )


@scenario("leakage_honest")
def leakage_honest(seed: int) -> Planted:
    """A target with strong but honest features, and independent ones."""
    rng = np.random.default_rng(seed)
    frame, target = _target_frame(rng, 2000)
    frame["strong"] = _signal(rng, target, 2.0)
    frame["flip_20pct"] = _flip(rng, target, 0.2)
    frame["many_levels"] = rng.choice([f"c{i}" for i in range(40)], len(target))
    return Planted(
        frame,
        plants=(
            plant("mutual_information", "signal"),
            plant("mutual_information", "strong"),
        ),
        allowed=(plant("mutual_information", "flip_20pct"),),
        target="target",
        checks=LEAK_CHECKS,
    )


@scenario("leakage_small_sample", seeds=60)
def leakage_small_sample(seed: int) -> Planted:
    """60 rows with a target and independent numeric and categorical features."""
    rng = np.random.default_rng(seed)
    n = 60
    target = (rng.random(n) < 0.4).astype(int)
    frame = pd.DataFrame(
        {
            "target": target,
            "noise_num": rng.normal(0, 1, n),
            "noise_cat": rng.choice(list("abcdefgh"), n),
            "noise_bin": rng.integers(0, 2, n),
        }
    )
    return Planted(frame, target="target", checks=LEAK_CHECKS)


@scenario("leakage_indirect")
def leakage_indirect(seed: int) -> Planted:
    """Leaks that show in the missing values, an identifier's prefix and a timestamp."""
    rng = np.random.default_rng(seed)
    n = 2000
    target = (rng.random(n) < 0.4).astype(int)
    opened = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n), "D")
    frame = pd.DataFrame(
        {
            "opened_at": opened,
            "target": target,
            "signal": _signal(rng, target, 1.0),
            "noise": rng.normal(0, 1, n),
            "null_when_negative": np.where(target == 1, rng.normal(0, 1, n), np.nan),
            "case_ref": [
                f"{'RES' if t else 'OPN'}-{rng.integers(1000, 9999)}" for t in target
            ],
            "closed_at": opened + pd.to_timedelta(rng.integers(5, 60, n), "D"),
            "reviewed_at": opened - pd.to_timedelta(rng.integers(5, 60, n), "D"),
        }
    )
    return Planted(
        frame,
        plants=(
            plant("target_leakage", "null_when_negative"),
            plant("target_leakage", "case_ref"),
            plant("target_leakage", "closed_at"),
        ),
        target="target",
        time_index="opened_at",
        checks=("target_leakage",),
    )
