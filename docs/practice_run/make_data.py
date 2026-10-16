"""Generate the practice-run datasets into ./data."""

import os

import numpy as np
import pandas as pd

from tests.generators.dsbf_test_dataset_generator import DSBFTestDataGenerator

os.makedirs("data", exist_ok=True)

for n in (5_000, 50_000):
    df = DSBFTestDataGenerator(n_rows=n, random_seed=42).generate()
    df.to_csv(f"data/kitchen_{n}.csv", index=False)
    print(f"kitchen_{n}", df.shape)

rng = np.random.default_rng(0)
n = 200
adv = pd.DataFrame(
    {
        "price|usd": rng.normal(100, 15, n),
        "cost|usd": rng.normal(100, 15, n),  # independent of price
        "[bold]": rng.normal(0, 1, n),
        "[/x]": rng.integers(0, 5, n),
        "naïve résumé 名前": rng.choice(["a", "b", "c"], n),
        "": rng.normal(0, 1, n),
        "all_null": [None] * n,
        "flag_as_int": rng.integers(0, 2, n),
        "flag_copy": None,
        "ts_tz": pd.date_range("2024-01-01", periods=n, freq="h", tz="US/Eastern"),
        "with,comma": rng.normal(0, 1, n),
        "inf_col": np.where(rng.random(n) < 0.05, np.inf, rng.normal(0, 1, n)),
    }
)
adv["flag_copy"] = adv["flag_as_int"]  # int-coded booleans, perfectly correlated
adv.to_csv("data/adversarial.csv", index=False)
print("adversarial", adv.shape)
