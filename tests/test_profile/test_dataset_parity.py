"""The dataset profile and stage agree with the committed legacy golden snapshots.

The snapshots hold what the legacy tasks reported on each golden dataset, so no legacy
task runs here. The CSV is read with ``load`` and profiled. Differences are listed in
``STAGES`` (and explained in ``accepted``).
"""

import json

import pytest

from dsbf.data import load
from dsbf.model import Stage
from dsbf.profiling import infer_stage, profile_dataset
from tests.golden.datasets import GOLDEN_DATASETS, GoldenDataset

pytestmark = pytest.mark.unit

# Legacy stage, new stage and the accepted difference behind any change of name. A
# dataset the two agree on carries no slug.
STAGES: dict[str, tuple[str, Stage, str | None]] = {
    "all_categorical": ("exploratory", Stage.RAW, "stage-names"),
    "clean": (
        "exploratory",
        Stage.CLEANED,
        "high-cardinality-counts-categorical-columns-only",
    ),
    "high_missingness": ("exploratory", Stage.RAW, "stage-names"),
    "kitchen5k": ("exploratory", Stage.RAW, "stage-names"),
    "near_clean": (
        "exploratory",
        Stage.CLEANED,
        "high-cardinality-counts-categorical-columns-only",
    ),
    "severe_multicollinearity": (
        "exploratory",
        Stage.CLEANED,
        "high-cardinality-counts-categorical-columns-only",
    ),
    "single_column": ("model_ready", Stage.MODELING, "stage-names"),
    "tiny": ("cleaned", Stage.CLEANED, None),
    "titanic": (
        "exploratory",
        Stage.CLEANED,
        "high-cardinality-counts-categorical-columns-only",
    ),
    "wide": ("exploratory", Stage.RAW, "stage-names"),
}


def _snapshot(dataset: GoldenDataset) -> dict:
    return json.loads(dataset.snapshot_path.read_text())


@pytest.mark.parametrize("dataset", GOLDEN_DATASETS, ids=lambda d: d.name)
def test_shape_duplicates_and_nulls_match_the_legacy_tasks(
    dataset: GoldenDataset, tmp_path
):
    tasks = _snapshot(dataset)["tasks"]
    shape = tasks["summarize_dataset_shape"]["data"]
    profile = profile_dataset(load(dataset.write_csv(tmp_path)))

    assert (profile.n_rows, profile.n_columns) == (
        shape["num_rows"],
        shape["num_columns"],
    )
    duplicates = tasks["detect_duplicates"]["data"]["duplicate_count"]
    assert profile.duplicate_rows == duplicates
    # The legacy "percentage" is a fraction of cells, rounded to four places.
    nulls = sum(c.counts.nulls for c in profile.columns)
    cells = profile.n_rows * profile.n_columns
    assert round(nulls / cells, 4) == shape["null_cell_percentage"]


@pytest.mark.parametrize("dataset", GOLDEN_DATASETS, ids=lambda d: d.name)
def test_the_stage_is_the_expected_one_for_each_dataset(
    dataset: GoldenDataset, tmp_path
):
    legacy, expected, _ = STAGES[dataset.name]
    profile = profile_dataset(load(dataset.write_csv(tmp_path)))

    assert _snapshot(dataset)["run"]["inferred_stage"] == legacy
    assert infer_stage(profile) is expected


def test_every_dataset_has_an_expected_stage():
    assert set(STAGES) == {d.name for d in GOLDEN_DATASETS}


def test_each_stage_difference_names_an_accepted_slug():
    from tests.test_profile.accepted import BY_SLUG

    names = {"model_ready": Stage.MODELING, "exploratory": Stage.RAW}
    for legacy, new, slug in STAGES.values():
        renamed_only = names.get(legacy) is new
        assert (slug is None) == (legacy == new.value)
        if slug is not None:
            assert slug in BY_SLUG
            # A change that is only a name needs "stage-names"; any other change
            # needs a rule.
            assert (slug == "stage-names") == renamed_only
