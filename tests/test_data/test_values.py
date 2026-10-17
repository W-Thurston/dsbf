# tests/test_data/test_values.py
"""Dataset.values returns cached read-only arrays within a memory bound."""

import datetime as dt
import threading

import numpy as np
import polars as pl
import pytest

from dsbf.data import load
from dsbf.data.values import ValueCache

pytestmark = pytest.mark.unit


def _numbers():
    return load(
        pl.DataFrame(
            {
                "i": [1, None, 3],
                "f": [1.5, float("nan"), None],
                "inf": [float("inf"), 1.0, float("nan")],
                "b": [True, None, False],
                "s": ["a", None, "c"],
            }
        )
    )


def test_values_drop_nulls_and_nan_by_default():
    dataset = _numbers()

    assert dataset.values("i").tolist() == [1, 3]
    assert dataset.values("i").dtype == np.int64
    assert dataset.values("f").tolist() == [1.5]
    assert dataset.values("b").tolist() == [True, False]
    assert dataset.values("s").tolist() == ["a", "c"]


def test_infinity_is_a_value_not_a_missing_one():
    assert _numbers().values("inf").tolist() == [float("inf"), 1.0]


def test_dropna_false_keeps_every_row():
    dataset = _numbers()

    integers = dataset.values("i", dropna=False)
    assert integers.dtype == np.float64
    assert np.isnan(integers[1])
    assert dataset.values("f", dropna=False).shape == (3,)
    assert dataset.values("b", dropna=False).tolist() == [True, None, False]
    assert dataset.values("s", dropna=False).tolist() == ["a", None, "c"]


def test_other_dtypes_come_back_as_arrays():
    dataset = load(
        pl.DataFrame(
            {
                "when": [dt.datetime(2024, 1, 1), None],
                "cat": pl.Series(["x", None], dtype=pl.Categorical),
                "tags": [[1, 2], None],
                "pair": [{"a": 1, "b": "x"}, None],
            }
        )
    )

    assert dataset.values("when").dtype == np.dtype("datetime64[us]")
    assert dataset.values("cat").tolist() == ["x"]
    assert [list(v) for v in dataset.values("tags")] == [[1, 2]]
    assert dataset.values("pair").tolist() == [{"a": 1, "b": "x"}]


def test_the_same_array_comes_back_until_evicted():
    dataset = _numbers()

    assert dataset.values("i") is dataset.values("i")
    assert dataset.values("i") is not dataset.values("i", dropna=False)


def test_arrays_are_read_only():
    array = _numbers().values("i")

    with pytest.raises(ValueError, match="read-only"):
        array[0] = 99


def test_an_unknown_column_is_a_key_error():
    with pytest.raises(KeyError, match="nope"):
        _numbers().values("nope")


def test_the_cache_is_per_dataset():
    one, two = _numbers(), _numbers()

    assert one.values("i") is not two.values("i")


# --- the memory bound ---------------------------------------------------------------


def _wide(n_columns=4, rows=1000, **kwargs):
    frame = pl.DataFrame(
        {f"c{i}": np.arange(rows, dtype=np.float64) for i in range(n_columns)}
    )
    return load(frame).__class__(
        frame=frame, name="w", source=load(frame).source, **kwargs
    )


def test_the_least_recently_used_array_is_evicted_first():
    dataset = _wide(value_cache_bytes=2 * 8000)  # room for two 1000-row columns
    first = dataset.values("c0")
    dataset.values("c1")
    dataset.values("c0")  # c0 is now the more recent
    dataset.values("c2")  # evicts c1

    assert dataset.values("c0") is first
    assert dataset._state.values.bytes_used == 2 * 8000
    assert dataset.values("c1") is not None  # recomputed, evicting c2 or c0
    assert dataset._state.values.bytes_used <= 2 * 8000


def test_the_bound_holds_however_many_columns_are_read():
    dataset = _wide(n_columns=10, value_cache_bytes=3 * 8000)

    for name in dataset.columns * 2:
        dataset.values(name)
        assert dataset._state.values.bytes_used <= 3 * 8000
    assert len(dataset._state.values) == 3


def test_an_array_larger_than_the_bound_is_returned_but_not_kept():
    dataset = _wide(value_cache_bytes=100)

    array = dataset.values("c0")

    assert len(array) == 1000
    assert len(dataset._state.values) == 0
    assert dataset.values("c0") is not array


def test_a_zero_bound_disables_the_cache():
    dataset = _wide(value_cache_bytes=0)

    assert dataset.values("c0") is not dataset.values("c0")


def test_a_negative_bound_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        _wide(value_cache_bytes=-1)


def test_text_is_charged_for_its_strings_not_just_its_pointers():
    frame = pl.DataFrame({"s": ["x" * 1000] * 100})
    dataset = load(frame)

    dataset.values("s")

    assert dataset._state.values.bytes_used > 100 * 1000


def test_default_bound_is_256_mib():
    assert load(pl.DataFrame({"a": [1]})).value_cache_bytes == 256 * 1024 * 1024


def test_cache_put_replaces_an_entry_without_double_counting():
    cache = ValueCache(1000)
    key = ("a", True)

    cache.put(key, np.zeros(10), 80)
    cache.put(key, np.zeros(20), 160)

    assert cache.bytes_used == 160
    assert len(cache) == 1


def test_threads_reading_together_get_correct_arrays():
    dataset = _wide(n_columns=8, value_cache_bytes=3 * 8000)
    failures = []

    def read(seed):
        rng = np.random.default_rng(seed)
        for _ in range(200):
            name = f"c{rng.integers(0, 8)}"
            if dataset.values(name).sum() != 499500.0:
                failures.append(name)

    threads = [threading.Thread(target=read, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert dataset._state.values.bytes_used <= 3 * 8000


def test_renaming_a_dataset_keeps_it_working():
    renamed = load(_numbers(), name="other")

    assert renamed.name == "other"
    assert renamed.values("i").tolist() == [1, 3]
