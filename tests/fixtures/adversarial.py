# tests/fixtures/adversarial.py
"""A small frame of hostile but legal data that every task has to survive."""

import numpy as np
import pandas as pd

N_ROWS = 200


def build_adversarial_frame(n_rows: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Return a frame that trips the usual per-column and pairwise assumptions.

    Column names contain ``|``, brackets, a comma, non-ASCII text, or are empty.
    ``all_null`` (object) and ``all_nan_float`` (float, the dtype a CSV round trip
    gives) have no values, ``inf_col`` has non-finite values, ``ts_tz`` is a
    timezone-aware datetime, and ``flag_as_int``/``flag_copy`` are identical
    integer-coded booleans. ``price|eur`` is a rescaled copy of ``price|usd``, so
    the leakage scan flags a pair whose names contain the ``|`` pair delimiter.
    """
    rng = np.random.default_rng(seed)
    flag = rng.integers(0, 2, n_rows)
    price = rng.normal(100, 15, n_rows)
    return pd.DataFrame(
        {
            "price|usd": price,
            "price|eur": price * 0.9,
            "cost|usd": rng.normal(100, 15, n_rows),
            "[bold]": rng.normal(0, 1, n_rows),
            "[/x]": rng.integers(0, 5, n_rows),
            "naïve résumé 名前": rng.choice(["a", "b", "c"], n_rows),
            "": rng.normal(0, 1, n_rows),
            "all_null": [None] * n_rows,
            "all_nan_float": np.full(n_rows, np.nan),
            "flag_as_int": flag,
            "flag_copy": flag,
            "ts_tz": pd.date_range(
                "2024-01-01", periods=n_rows, freq="h", tz="US/Eastern"
            ),
            "with,comma": rng.normal(0, 1, n_rows),
            "inf_col": np.where(
                rng.random(n_rows) < 0.05, np.inf, rng.normal(0, 1, n_rows)
            ),
        }
    )
