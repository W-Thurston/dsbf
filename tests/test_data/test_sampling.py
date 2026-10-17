# tests/test_data/test_sampling.py
"""Dataset.sample draws reproducibly and records how."""

import numpy as np
import polars as pl
import pytest

from dsbf.data import Dataset, SamplingError, load
from dsbf.model.evidence import SamplingInfo

pytestmark = pytest.mark.unit

N = 1000
GROUPS = ["a"] * 700 + ["b"] * 250 + ["c"] * 45 + ["d"] * 4 + [None]
FRAME = pl.DataFrame(
    {
        "row": np.arange(N),
        "group": GROUPS,
        "other": ["u", "v"] * (N // 2),
        "x": np.random.default_rng(1).normal(size=N),
    }
)


def _dataset():
    return load(FRAME)


def _info(dataset: Dataset) -> SamplingInfo:
    assert dataset.sampling is not None
    return dataset.sampling


def _rows(dataset: Dataset) -> np.ndarray:
    return dataset.frame["row"].to_numpy()


def test_head_takes_the_first_rows_and_records_it():
    sample = _dataset().sample(10)

    assert sample.frame["row"].to_list() == list(range(10))
    assert sample.sampling == SamplingInfo(
        strategy="head", original_rows=N, sampled_rows=10
    )


@pytest.mark.parametrize("strategy", ["random", "stratified"])
def test_a_seeded_sample_is_reproducible(strategy):
    kwargs = {
        "strategy": strategy,
        "stratify_by": ["group"] if "s" == strategy[0] else [],
    }

    first = _dataset().sample(100, seed=5, **kwargs)
    again = _dataset().sample(100, seed=5, **kwargs)
    other = _dataset().sample(100, seed=6, **kwargs)

    assert first.frame.equals(again.frame)
    assert not first.frame.equals(other.frame)
    assert _info(first).seed == 5


def test_random_defaults_to_seed_zero_and_says_so():
    sample = _dataset().sample(50, strategy="random")

    assert _info(sample).seed == 0
    assert sample.frame.equals(_dataset().sample(50, strategy="random", seed=0).frame)


def test_random_has_distinct_rows_in_their_original_order():
    rows = _dataset().sample(300, strategy="random", seed=2).frame["row"]

    assert rows.n_unique() == 300
    assert rows.is_sorted()
    assert rows.to_numpy().max() > 900  # not just a head


def test_random_is_not_biased_toward_the_front():
    rows = _dataset().sample(500, strategy="random", seed=3).frame["row"].to_numpy()

    assert abs(rows.mean() - (N - 1) / 2) < 50


def test_stratified_has_exactly_the_requested_rows_in_proportion():
    sample = _dataset().sample(
        100, strategy="stratified", stratify_by=["group"], seed=0
    )
    counts = dict(sample.frame["group"].value_counts().iter_rows())

    assert sample.n_rows == 100
    assert counts["a"] == 70
    assert counts["b"] == 25
    assert counts["c"] == 5  # 4.5 rounds up: the largest remainder
    assert _info(sample).stratify_by == ("group",)
    assert sample.frame["row"].is_sorted()


@pytest.mark.parametrize("rows", [1, 2, 7, 33, 100, 999])
def test_stratified_total_is_exact_for_any_size(rows):
    sample = _dataset().sample(rows, strategy="stratified", stratify_by=["group"])

    assert sample.n_rows == rows
    assert _info(sample).sampled_rows == rows


def test_stratified_treats_null_as_its_own_group():
    frame = pl.DataFrame({"g": [None] * 50 + ["a"] * 50, "i": range(100)})

    sample = load(frame).sample(10, strategy="stratified", stratify_by=["g"])
    counts = dict(sample.frame["g"].value_counts().iter_rows())

    assert counts == {None: 5, "a": 5}


def test_stratified_on_several_columns():
    sample = _dataset().sample(
        200, strategy="stratified", stratify_by=["group", "other"], seed=1
    )

    assert sample.n_rows == 200
    assert _info(sample).stratify_by == ("group", "other")


def test_stratified_is_not_head_in_disguise():
    sample = _dataset().sample(50, strategy="stratified", stratify_by=["other"], seed=4)

    assert _rows(sample).max() > 500


def test_a_sample_keeps_its_sources_fingerprint():
    dataset = _dataset()
    sample = dataset.sample(10)

    assert sample.fingerprint == dataset.fingerprint
    assert sample.source_rows == N
    assert sample.n_rows == 10
    assert dataset.source_rows == N


def test_asking_for_all_the_rows_is_no_sample():
    dataset = _dataset()

    assert dataset.sample(N) is dataset
    assert dataset.sample(N * 2, strategy="random") is dataset
    assert dataset.sampling is None


def test_a_sample_has_its_own_values_and_name():
    dataset = load(FRAME, name="t")
    sample = dataset.sample(10)

    assert sample.name == "t"
    assert len(sample.values("row")) == 10
    assert len(dataset.values("row")) == N


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"rows": 0}, "positive integer"),
        ({"rows": -3}, "positive integer"),
        ({"rows": 2.5}, "positive integer"),
        ({"rows": True}, "positive integer"),
        ({"rows": 5, "strategy": "bootstrap"}, "Unknown sampling strategy"),
        ({"rows": 5, "strategy": "stratified"}, "needs stratify_by"),
        (
            {"rows": 5, "strategy": "stratified", "stratify_by": ["nope"]},
            "missing column",
        ),
        ({"rows": 5, "stratify_by": ["group"]}, "applies to the stratified"),
        ({"rows": 5, "strategy": "random", "stratify_by": ["group"]}, "applies to"),
        ({"rows": 5, "seed": 3}, "no seed"),
    ],
)
def test_bad_requests_raise_instead_of_falling_back(kwargs, message):
    with pytest.raises(SamplingError, match=message):
        _dataset().sample(**kwargs)


def test_a_bad_request_is_an_error_even_when_the_table_is_small_enough():
    with pytest.raises(SamplingError, match="missing column"):
        _dataset().sample(N * 2, strategy="stratified", stratify_by=["nope"])


def test_a_sample_cannot_be_sampled_again():
    with pytest.raises(SamplingError, match="already a sample"):
        _dataset().sample(100).sample(10)


def test_sampling_error_is_a_value_error():
    assert issubclass(SamplingError, ValueError)


def test_sampling_does_not_touch_the_source():
    dataset = _dataset()
    before = dataset.frame.clone()

    dataset.sample(10, strategy="random")

    assert dataset.frame.equals(before)
