import pytest

from dsbf.store.sketch import MAX_BINS, _recut

pytestmark = pytest.mark.unit


def test_recut_keeps_the_total_and_the_bin_limit():
    counts = tuple(range(1, 38))
    recut = _recut(counts)
    assert len(recut) == MAX_BINS
    assert sum(recut) == sum(counts)


def test_recut_leaves_a_coarse_histogram_alone():
    assert _recut((3, 0, 5)) == (3, 0, 5)
