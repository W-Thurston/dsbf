"""The rule checks cost next to nothing, because they read the profile."""

import pytest

from dsbf.checks import core_pack
from dsbf.data import load
from dsbf.engine import Registry, Selection, execute, make_plan
from dsbf.profiling import profile_dataset
from tests.generators.dsbf_test_dataset_generator import DSBFTestDataGenerator

pytestmark = [pytest.mark.integration, pytest.mark.slow]

COLUMN_HEALTH = (
    "null_rate",
    "constant_columns",
    "high_cardinality",
    "id_columns",
    "dominant_value",
    "near_zero_variance",
    "duplicate_rows",
    "duplicate_columns",
    "text_encoding",
)
SHAPE_AND_TYPES = (
    "skewness",
    "kurtosis",
    "entropy",
    "dtype_optimizations",
    "sample_size",
    "class_imbalance",
)
BUDGET_SECONDS = 0.05


@pytest.fixture(scope="module")
def kitchen50k():
    frame = DSBFTestDataGenerator(n_rows=50_000, random_seed=42).generate()
    dataset = load(frame, name="kitchen50k")
    return dataset, profile_dataset(dataset)


def total_seconds(kitchen50k, check_ids: tuple[str, ...], **params) -> float:
    dataset, profile = kitchen50k
    plan = make_plan(
        Registry(core_pack().checks),
        Selection(include=check_ids),
        params=params or None,
        profile=profile,
    )

    run = execute(plan, dataset, profile, dsbf_version="test", workers=1)

    results = [run.results[check_id] for check_id in check_ids]
    assert [r.status.value for r in results] == ["ok"] * len(check_ids)
    total = sum(r.timing.wall_seconds for r in results)
    assert total < BUDGET_SECONDS, {
        r.check_id: round(r.timing.wall_seconds, 4) for r in results
    }
    return total


def test_the_nine_checks_take_under_50_ms_together_on_kitchen50k(kitchen50k) -> None:
    total_seconds(kitchen50k, COLUMN_HEALTH)


def test_the_six_shape_and_type_checks_take_under_50_ms_on_kitchen50k(
    kitchen50k,
) -> None:
    total_seconds(
        kitchen50k, SHAPE_AND_TYPES, class_imbalance={"target": "balanced_target"}
    )
