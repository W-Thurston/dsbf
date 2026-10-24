"""The fuzzy duplicate check finds the near-duplicates planted in the test frame."""

import json

import pytest

from dsbf.checks import core_pack
from dsbf.data import load
from dsbf.engine import Registry, Selection, execute, make_plan
from dsbf.model import CheckStatus
from dsbf.profiling import profile_dataset
from tests.generators.dsbf_test_dataset_generator import DSBFTestDataGenerator
from tests.golden.datasets import DATA_DIR

pytestmark = [pytest.mark.integration, pytest.mark.slow]

PLANTED = 50
# Legacy `fuzzy_duplicate_detection` took 68 s on a 5 000-row frame of this kind
# (CI runner, U4-01); the check must stay well under a quarter of that.
SECONDS = 15.0


def run_fuzzy(source, **params):
    dataset = load(source, name="frame")
    profile = profile_dataset(dataset)
    plan = make_plan(
        Registry(core_pack().checks),
        Selection(include="fuzzy_duplicates"),
        params={"fuzzy_duplicates": params} if params else None,
        profile=profile,
    )
    run = execute(plan, dataset, profile, dsbf_version="test", workers=1)
    return run.results["fuzzy_duplicates"]


@pytest.fixture(scope="module")
def planted():
    generator = DSBFTestDataGenerator(
        n_rows=5000, random_seed=42, near_duplicates=PLANTED
    )
    frame = generator.generate()
    return frame, generator.planted_near_duplicates


def test_every_planted_copy_is_found_and_nothing_else(planted):
    frame, copies = planted

    result = run_fuzzy(frame, max_pairs_listed=200)

    assert result.status is CheckStatus.OK
    assert result.timing.wall_seconds < SECONDS
    output = result.output
    assert output is not None
    found = {
        (p["first"]["row"], p["second"]["row"]): p["similarity"]
        for p in output["top_pairs"]
    }
    expected = {(source, copy): kind for source, copy, kind in copies}
    assert len(expected) == PLANTED
    # Recall 50 of 50 at the default budgets (rare-token blocking, not every pair).
    assert set(found) == set(expected)
    assert output["blocking"] == "rare_tokens" and output["stopped_by"] is None
    assert output["pairs_flagged"] == PLANTED
    assert min(found.values()) > 0.95
    assert [n.code for n in result.notes] == []


def test_the_frame_without_copies_has_no_pair(planted):
    frame, copies = planted

    result = run_fuzzy(frame.iloc[: len(frame) - PLANTED])

    assert result.output is not None and result.output["pairs_flagged"] == 0


def test_titanic_has_no_text_to_compare_and_the_result_is_small():
    result = run_fuzzy(DATA_DIR / "titanic.csv")

    assert result.status is CheckStatus.NOT_APPLICABLE
    assert len(json.dumps(result.model_dump(mode="json"))) < 100_000
