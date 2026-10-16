# tests/test_utils/test_data_utils.py

import numpy as np
import pandas as pd
import pytest

from dsbf.utils.data_utils import data_sampling, ranked_value_counts

pytestmark = pytest.mark.unit


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


def test_ranked_value_counts_orders_by_count_then_first_appearance():
    series = pd.Series(["b", "a", "c", "a", "b", "d", "c"])
    ranked = ranked_value_counts(series)
    assert list(ranked.index) == ["b", "a", "c", "d"]
    assert list(ranked.to_numpy()) == [2, 2, 2, 1]


def test_ranked_value_counts_breaks_large_ties_by_first_appearance():
    # Every value occurs twice, so the tie-break decides the whole ranking.
    values = np.random.default_rng(0).permutation(np.repeat(np.arange(5000.0), 2))
    ranked = ranked_value_counts(pd.Series(values))
    first_seen = list(dict.fromkeys(values))
    assert list(ranked.index) == first_seen


def test_ranked_value_counts_keeps_nulls_only_when_asked():
    series = pd.Series([1.0, None, None, 2.0])
    assert ranked_value_counts(series).sum() == 2
    with_nulls = ranked_value_counts(series, dropna=False)
    assert with_nulls.iloc[0] == 2
    assert pd.isna(with_nulls.index[0])
