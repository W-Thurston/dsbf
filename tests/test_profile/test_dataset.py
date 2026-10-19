"""The dataset profile: shape, duplicate rows and which columns go missing together."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.data import load
from dsbf.model import DatasetProfile, Missingness
from dsbf.profiling import profile_dataset
from tests.fixtures.adversarial import N_ROWS, build_adversarial_frame

pytestmark = pytest.mark.unit


def _profile(frame, **kwargs) -> DatasetProfile:
    return profile_dataset(load(frame), **kwargs)


def _missingness(frame, **kwargs) -> Missingness:
    missingness = _profile(frame, **kwargs).missingness
    assert missingness is not None
    return missingness


def test_shape_and_memory_are_the_profiled_frames():
    frame = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    profile = _profile(frame)

    assert profile.n_rows == 3
    assert profile.n_columns == 2
    assert profile.approx_memory_bytes == frame.estimated_size() > 0
    assert [c.name for c in profile.columns] == ["a", "b"]


def test_a_sample_profiles_its_own_rows():
    dataset = load(pl.DataFrame({"a": range(100)}))

    profile = profile_dataset(dataset.sample(10))

    assert profile.n_rows == 10


def test_duplicate_rows_are_rows_equal_to_an_earlier_row():
    frame = pl.DataFrame({"a": [1, 1, 1, 2], "b": ["x", "x", "x", "x"]})

    assert _profile(frame).duplicate_rows == 2


def test_duplicate_rows_treat_signed_zeros_nans_and_nulls_as_equal():
    frame = pl.DataFrame(
        {"a": [0.0, -0.0, float("nan"), float("nan"), None, None, 1.0]}
    )

    assert _profile(frame).duplicate_rows == 3


def test_duplicate_rows_of_list_cells():
    frame = pl.DataFrame({"a": [[1, 2], [1, 2], [3]]})

    assert _profile(frame).duplicate_rows == 1


def test_duplicate_rows_agree_with_pandas_on_a_wide_frame():
    rng = np.random.default_rng(1)
    base = pd.DataFrame(rng.integers(0, 3, (500, 4)), columns=list("abcd"))
    frame = pd.concat([base, base.iloc[:50]], ignore_index=True)

    assert _profile(frame).duplicate_rows == int(frame.duplicated().sum())


def test_rows_with_any_null_counts_rows_not_cells():
    frame = pl.DataFrame({"a": [1, None, None, 4], "b": [None, None, 3, 4]})

    assert _profile(frame).rows_with_any_null == 3


def test_a_polars_nan_is_a_value_not_a_null():
    frame = pl.DataFrame({"a": [1.0, float("nan"), None]})

    profile = _profile(frame)

    assert profile.rows_with_any_null == 1
    assert _missingness(frame).columns_with_nulls == 1


def test_a_pandas_nan_is_a_null():
    profile = _profile(pd.DataFrame({"a": [1.0, np.nan, 3.0]}))

    assert profile.rows_with_any_null == 1


def test_missingness_pairs_columns_that_are_null_together():
    frame = pl.DataFrame(
        {
            "a": [None, None, 1, 2, 3, 4],
            "b": [None, None, 5, 6, 7, 8],
            "c": [None, 9, 9, 9, 9, 9],
            "d": [1, 2, 3, 4, 5, 6],
        }
    )

    missingness = _missingness(frame)

    assert missingness.columns_with_nulls == 3
    assert missingness.columns_considered == 3
    assert [(p.a, p.b, p.both_null, p.jaccard) for p in missingness.pairs] == [
        ("a", "b", 2, 1.0),
        ("a", "c", 1, 0.5),
        ("b", "c", 1, 0.5),
    ]


def test_missingness_pairs_match_a_pandas_crosstab():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(2000, 6)), columns=list("abcdef"))
    frame = frame.mask(rng.random(frame.shape) < 0.2)
    nulls = frame.isna()

    missingness = _missingness(frame, max_pairs=100)

    assert len(missingness.pairs) == 15
    for pair in missingness.pairs:
        both = int((nulls[pair.a] & nulls[pair.b]).sum())
        either = int((nulls[pair.a] | nulls[pair.b]).sum())
        assert pair.both_null == both
        assert pair.jaccard == pytest.approx(both / either)
    scores = [p.jaccard for p in missingness.pairs]
    assert scores == sorted(scores, reverse=True)


def test_pairs_that_share_no_null_row_are_left_out():
    frame = pl.DataFrame({"a": [None, 1, 2], "b": [1, None, 2]})

    missingness = _missingness(frame)

    assert missingness.columns_with_nulls == 2
    assert missingness.pairs == ()


def test_missingness_of_complete_data_is_empty():
    missingness = _missingness(pl.DataFrame({"a": [1, 2]}))

    assert (missingness.columns_with_nulls, missingness.columns_considered) == (0, 0)
    assert missingness.pairs == ()


def test_only_the_columns_with_most_nulls_are_paired():
    frame = pl.DataFrame(
        {
            "few": [None, 1, 1, 1, 1, 1],
            "most": [None, None, None, None, 1, 1],
            "more": [None, None, None, 1, 1, 1],
            "all_present": [1, 1, 1, 1, 1, 1],
        }
    )

    missingness = _missingness(frame, max_pair_columns=2)

    assert missingness.columns_with_nulls == 3
    assert missingness.columns_considered == 2
    assert [(p.a, p.b) for p in missingness.pairs] == [("most", "more")]


def test_the_pair_list_is_capped_at_max_pairs():
    frame = pl.DataFrame({name: [None, 1] for name in "abcd"})

    assert len(_missingness(frame, max_pairs=2).pairs) == 2
    assert _missingness(frame, max_pairs=0).pairs == ()


def test_the_null_mask_is_chunked_without_changing_counts(monkeypatch):
    monkeypatch.setattr("dsbf.profiling.dataset._MASK_CHUNK_BYTES", 4 * 2 * 7)
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
    frame = frame.mask(rng.random(frame.shape) < 0.3)

    pairs = _missingness(frame, max_pairs=10).pairs

    nulls = frame.isna()
    assert {(p.a, p.b): p.both_null for p in pairs} == {
        (a, b): int((nulls[a] & nulls[b]).sum())
        for a, b in [("a", "b"), ("a", "c"), ("b", "c")]
    }


@pytest.mark.parametrize("name", ["max_pair_columns", "max_pairs"])
def test_negative_limits_are_refused(name):
    with pytest.raises(ValueError, match="negative"):
        _profile(pl.DataFrame({"a": [1]}), **{name: -1})


def test_the_adversarial_frame_profiles_and_round_trips():
    frame = build_adversarial_frame()
    profile = _profile(frame)
    missingness = _missingness(frame)

    assert profile.n_rows == N_ROWS
    assert profile.duplicate_rows == 0
    assert DatasetProfile.model_validate_json(profile.model_dump_json()) == profile
    assert missingness.columns_with_nulls >= 2  # all_null, all_nan_float
    assert ("all_null", "all_nan_float") in {(p.a, p.b) for p in missingness.pairs}


def test_an_empty_frame_has_a_profile():
    frame = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
    profile = _profile(frame)

    assert profile.n_rows == 0
    assert profile.duplicate_rows == 0
    assert profile.rows_with_any_null == 0
    assert _missingness(frame).pairs == ()
