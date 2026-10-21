"""The column health checks cost next to nothing, because they read the profile."""

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
BUDGET_SECONDS = 0.05


def test_the_nine_checks_take_under_50_ms_together_on_kitchen50k() -> None:
    frame = DSBFTestDataGenerator(n_rows=50_000, random_seed=42).generate()
    dataset = load(frame, name="kitchen50k")
    profile = profile_dataset(dataset)
    plan = make_plan(
        Registry(core_pack().checks),
        Selection(include=COLUMN_HEALTH),
        profile=profile,
    )

    run = execute(plan, dataset, profile, dsbf_version="test", workers=1)

    results = [run.results[check_id] for check_id in COLUMN_HEALTH]
    assert [r.status.value for r in results] == ["ok"] * len(COLUMN_HEALTH)
    total = sum(r.timing.wall_seconds for r in results)
    assert total < BUDGET_SECONDS, {
        r.check_id: round(r.timing.wall_seconds, 4) for r in results
    }
