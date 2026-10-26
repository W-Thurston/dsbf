"""The sample table the guides and the example workflow profile."""

from pathlib import Path

import numpy as np
import polars as pl


def write_customers(path: Path) -> Path:
    """Write a 3 000-row customer table whose purchase amounts have heavy tails."""
    rng = np.random.default_rng(7)
    rows = 3000
    frame = pl.DataFrame(
        {
            "age": rng.integers(18, 80, rows),
            "income": rng.normal(60_000, 15_000, rows).round(2),
            "purchase_amount": rng.lognormal(3.4, 0.9, rows).round(2),
            "tenure_months": rng.integers(1, 120, rows),
            "satisfaction": rng.uniform(1, 10, rows).round(1),
            "region": rng.choice(["North", "South", "East", "West"], rows),
            "plan_type": rng.choice(["Basic", "Plus", "Premium"], rows),
            "segment": rng.choice(["SMB", "Enterprise", "Consumer"], rows),
            "is_churned": rng.random(rows) < 0.2,
        }
    )
    frame.write_csv(path)
    return path
