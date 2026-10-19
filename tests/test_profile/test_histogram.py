"""Histograms have equal-width bins, a recorded rule, and counts that add up."""

import math

import numpy as np
import polars as pl
import pytest

from dsbf.data import load
from dsbf.model.profile import NumericProfile
from dsbf.model.vocabulary import Confidence
from dsbf.profiling import ColumnType, profile_columns
from dsbf.profiling.histogram import histogram

pytestmark = pytest.mark.unit


def _histogram(values, dtype=None, **kwargs):
    return histogram(pl.Series(values, dtype=dtype), **kwargs)


class TestRule:
    def test_freedman_diaconis_sets_the_bin_width(self):
        values = np.random.default_rng(0).normal(size=4000)
        result = _histogram(values)
        assert result is not None
        q1, q3 = np.percentile(values, [25, 75])
        width = 2 * (q3 - q1) * values.size ** (-1 / 3)
        assert result.rule == "fd"
        assert len(result.counts) == math.ceil(np.ptp(values) / width)

    def test_a_zero_interquartile_range_falls_back_to_sturges(self):
        values = [0.0] * 90 + [float(v) for v in range(1, 11)]
        result = _histogram(values)
        assert result is not None
        assert result.rule == "sturges"
        assert len(result.counts) == math.ceil(math.log2(100)) + 1

    def test_a_heavy_tail_is_clipped_to_the_bin_limit_and_keeps_its_rule(self):
        values = [*np.random.default_rng(0).normal(size=1000), 1e12]
        result = _histogram(values)
        assert result is not None
        assert (result.rule, len(result.counts)) == ("fd", 100)
        narrower = _histogram(values, max_bins=20)
        assert narrower is not None
        assert len(narrower.counts) == 20

    def test_a_single_bin_is_allowed(self):
        result = _histogram([1.0, 2.0, 3.0], max_bins=1)
        assert result is not None
        assert result.counts == (3,)
        assert (result.edges[0], result.edges[-1]) == (1.0, 3.0)


class TestBins:
    def test_bins_are_equal_width_and_counts_add_up(self):
        values = np.random.default_rng(1).exponential(size=5000)
        result = _histogram(values)
        assert result is not None
        widths = np.diff(result.edges)
        assert np.allclose(widths, widths[0])
        assert result.total == values.size

    def test_counts_match_numpy_on_the_same_edges(self):
        values = np.random.default_rng(2).normal(size=3000)
        result = _histogram(values)
        assert result is not None
        expected, _ = np.histogram(values, bins=np.array(result.edges))
        assert result.counts == tuple(expected.tolist())

    def test_bins_are_closed_on_the_left_and_the_last_on_both_sides(self):
        result = _histogram([0.0, 1.0, 2.0, 3.0, 4.0])
        assert result is not None
        assert result.edges == (0.0, 2.0, 4.0)
        # 2.0 sits on the inner edge, so it opens the second bin; 4.0 closes it.
        assert result.counts == (2, 3)

    def test_uniform_data_gives_even_bins(self):
        # The F-12 defect: percentile bins made a uniform column look peaked.
        result = _histogram(np.linspace(0, 1, 10_001))
        assert result is not None
        assert max(result.counts) - min(result.counts) <= 1

    def test_integers_get_whole_number_bins(self):
        result = _histogram(list(range(10)) * 200)
        assert result is not None
        assert result.edges == tuple(float(v) for v in range(11))
        assert result.counts == (200,) * 10

    def test_wide_integer_ranges_use_a_whole_number_width(self):
        result = _histogram(list(range(0, 100_000, 7)))
        assert result is not None
        assert all(float(w).is_integer() for w in np.diff(result.edges))
        assert result.total == len(range(0, 100_000, 7))

    def test_negative_values_are_binned(self):
        result = _histogram([-5.0, -1.0, 0.0, 3.0, 8.0])
        assert result is not None
        assert result.edges[0] == -5.0
        assert result.total == 5

    def test_a_range_too_small_for_its_magnitude_still_has_increasing_edges(self):
        result = _histogram([1e16, 1e16 + 2, 1e16 + 4, 1e16 + 4], max_bins=100)
        assert result is not None
        assert result.total == 4

    @pytest.mark.parametrize(
        "dtype", [pl.UInt8, pl.Int8, pl.Int64, pl.UInt64, pl.Int128, pl.Float32]
    )
    def test_every_numeric_dtype_bins(self, dtype):
        result = _histogram([1, 2, 3, 4, 5, 6, 7, 8], dtype)
        assert result is not None
        assert result.total == 8


class TestNoHistogram:
    @pytest.mark.parametrize(
        "values",
        [[], [None, None], [5.0], [5.0, 5.0, 5.0], [float("nan"), float("inf")]],
    )
    def test_a_column_without_spread_has_none(self, values):
        assert _histogram(values, pl.Float64) is None

    def test_an_unrepresentable_range_has_none(self):
        assert _histogram([-1e308, 1e308]) is None


class TestInProfile:
    def _profile(self, values, dtype=None, **kwargs) -> NumericProfile:
        frame = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
        (profile,) = profile_columns(load(frame), **kwargs)
        assert isinstance(profile, NumericProfile)
        return profile

    def test_counts_add_up_to_the_non_null_values(self):
        profile = self._profile([*np.random.default_rng(3).normal(size=500), None])
        assert profile.histogram is not None
        assert profile.histogram.total == profile.counts.non_null == 500

    def test_non_finite_values_stay_out_of_the_bins(self):
        profile = self._profile([1.0, 2.0, 3.0, float("nan"), float("inf")], pl.Float64)
        assert profile.non_finite == 2
        assert profile.histogram is not None
        assert profile.histogram.total == profile.counts.non_null - 2

    def test_quartiles_come_from_the_requested_quantiles_when_present(self):
        values = np.random.default_rng(4).normal(size=2000)
        with_quartiles = self._profile(values)
        without = self._profile(values, quantiles=(0.5,))
        assert with_quartiles.histogram == without.histogram

    def test_the_bin_limit_is_validated(self):
        with pytest.raises(ValueError, match="max_bins"):
            self._profile([1, 2, 3], max_bins=0)

    def test_a_constant_column_has_no_histogram(self):
        frame = pl.DataFrame({"x": [4.0] * 30})
        forced = {"x": ColumnType("numeric", Confidence.HIGH, "Float64", "forced")}
        (profile,) = profile_columns(load(frame), forced)
        assert isinstance(profile, NumericProfile)
        assert profile.histogram is None
