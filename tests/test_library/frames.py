"""A small mixed frame for the end-to-end library tests."""

import datetime as dt

import numpy as np
import polars as pl

N_ROWS = 240


def build_frame() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    age = rng.normal(40, 12, N_ROWS)
    return pl.DataFrame(
        {
            "age": [None if i % 9 == 0 else float(v) for i, v in enumerate(age)],
            "fare": rng.lognormal(3, 1, N_ROWS),
            "group": rng.choice(["a", "b", "c"], N_ROWS),
            "flag": rng.integers(0, 2, N_ROWS).astype(bool),
            "when": [
                dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(N_ROWS)
            ],
            "trend": np.arange(N_ROWS) * 0.5 + rng.normal(0, 3, N_ROWS),
            "comment": [
                f"the quick brown fox jumps over lazy dog number {i % 70}"
                for i in range(N_ROWS)
            ],
            "row_id": [f"ID{i:05d}" for i in range(N_ROWS)],
        }
    )
