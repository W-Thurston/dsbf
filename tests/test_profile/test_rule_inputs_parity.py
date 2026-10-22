"""Entropy and column hashes agree with the committed legacy golden snapshots.

``compute_entropy`` and ``detect_duplicate_columns`` are compared by what they
reported on each golden dataset. The same columns have an entropy, within the six
significant figures the snapshots keep, and the columns the legacy task called
duplicates are the ones whose hash and dtype match. The one accepted difference is a
column of prose: the legacy types called it an identifier and measured no entropy, and
the profile types it `text` and does.
"""

import itertools
import json

import pytest

from dsbf.data import load
from dsbf.profiling import profile_dataset
from tests.golden.datasets import GOLDEN_DATASETS, GoldenDataset

pytestmark = pytest.mark.unit


def _tasks(dataset: GoldenDataset) -> dict:
    return json.loads(dataset.snapshot_path.read_text())["tasks"]


@pytest.mark.parametrize("dataset", GOLDEN_DATASETS, ids=lambda d: d.name)
def test_entropy_matches_compute_entropy(dataset: GoldenDataset, tmp_path):
    legacy = _tasks(dataset)["compute_entropy"]["data"]
    profile = profile_dataset(load(dataset.write_csv(tmp_path)))

    profiled = {
        column.name: column.entropy_bits
        for column in profile.columns
        if getattr(column, "entropy_bits", None) is not None
    }

    prose = {
        column.name for column in profile.columns if column.semantic_type == "text"
    }
    assert set(profiled) - set(legacy) <= prose
    assert set(legacy) <= set(profiled)
    for name, bits in legacy.items():
        assert profiled[name] == pytest.approx(bits, rel=1e-5, abs=1e-6), name


@pytest.mark.parametrize("dataset", GOLDEN_DATASETS, ids=lambda d: d.name)
def test_equal_hashes_and_dtypes_are_the_legacy_duplicate_pairs(
    dataset: GoldenDataset, tmp_path
):
    legacy = _tasks(dataset)["detect_duplicate_columns"]["data"]
    profile = profile_dataset(load(dataset.write_csv(tmp_path)))

    pairs = {
        frozenset((a.name, b.name))
        for a, b in itertools.combinations(profile.columns, 2)
        if a.content_hash is not None
        and a.content_hash == b.content_hash
        and a.physical_dtype == b.physical_dtype
    }

    assert pairs == {frozenset(pair) for pair in legacy["duplicate_column_pairs"]}


def test_kitchen5k_has_generated_duplicate_columns_with_equal_hashes(tmp_path):
    dataset = next(d for d in GOLDEN_DATASETS if d.name == "kitchen5k")
    profile = profile_dataset(load(dataset.write_csv(tmp_path)))
    hashes = {c.name: c.content_hash for c in profile.columns}

    assert (
        hashes["duplicate_col_1"]
        == hashes["duplicate_col_2"]
        == hashes["duplicate_col_3"]
    )
    assert hashes["duplicate_col_1"] != hashes["partial_duplicate_1"]
