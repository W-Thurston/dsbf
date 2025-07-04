# tests/test_utils/test_data_utils.py

import pandas as pd
import pytest

from dsbf.utils.data_utils import data_sampling


@pytest.mark.parametrize(
    "n_rows,strategy,expected_rows",
    [
        (500_000, "head", 500_000),  # Below threshold: no sampling
        (2_000_000, "head", 1_000_000),  # Above threshold: head sampling
        (2_000_000, "random", 1_000_000),  # Random sampling
        (2_000_000, "invalid_strategy", 1_000_000),  # Fallback to head
    ],
)
def test_data_sampling_behavior(n_rows, strategy, expected_rows):
    df = pd.DataFrame({"x": range(n_rows)})
    config = {
        "resource_limits": {
            "enable_sampling": True,
            "sample_threshold_rows": 1_000_000,
            "sample_strategy": strategy,
        }
    }

    sampled_df, info = data_sampling(df, config)
    assert len(sampled_df) == expected_rows

    if n_rows > 1_000_000:
        assert info is not None
        assert info["original_rows"] == n_rows
        assert info["sampled_rows"] == 1_000_000
        assert "strategy" in info
    else:
        assert info is None
