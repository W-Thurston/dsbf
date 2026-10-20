"""The datasets that are timed, and how to materialise each."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from tests.fixtures.adversarial import build_adversarial_frame
from tests.generators.dsbf_test_dataset_generator import DSBFTestDataGenerator
from tests.golden.datasets import KITCHEN5K, TITANIC

KITCHEN_SEED = 42
KITCHEN_TIME_SERIES_COLUMN = "normal_date"
PER_CORE: Literal["per-core"] = "per-core"


@dataclass(frozen=True)
class Scenario:
    """One timed dataset: a name, a writer for its CSV, and how often to repeat it.

    A run that takes seconds is repeated and reported by its median total, so one
    noisy run doesn't decide the result. Runs that take minutes are measured once.
    """

    name: str
    write_csv: Callable[[Path], Path]
    time_series_column: str | None = None
    repeats: int = 1


@dataclass(frozen=True)
class ProfileScenario:
    """One timed profile pass: a name, a writer for its data file, and the repeats.

    Unlike ``Scenario`` it never runs the legacy engine, so a large frame is cheap
    enough to repeat.
    """

    name: str
    write_data: Callable[[Path], Path]
    repeats: int = 3


@dataclass(frozen=True)
class EngineScenario:
    """One timed ``dsbf.profile()`` call: the new engine over the adapted legacy tasks.

    ``workers`` is how many checks run at once: a number, ``PER_CORE`` for the
    executor's default (one per core, at most 8), or ``None`` for what
    ``dsbf.profile()`` chooses when it is not given. A serial and a per-core scenario
    on the same data show what the parallel executor buys, so the pair is the
    measurement. ``depth`` is the plan's
    depth: ``deep`` runs every check, ``standard`` is what ``dsbf.profile()`` runs
    when no depth is given.
    """

    name: str
    write_csv: Callable[[Path], Path]
    workers: int | Literal["per-core"] | None
    time_series_column: str | None = None
    repeats: int = 1
    depth: str = "deep"


def _adversarial(directory: Path) -> Path:
    path = directory / "adversarial.csv"
    build_adversarial_frame().to_csv(path, index=False)
    return path


def _kitchen(rows: int, name: str) -> Callable[[Path], Path]:
    def write_csv(directory: Path) -> Path:
        path = directory / f"{name}.csv"
        generator = DSBFTestDataGenerator(n_rows=rows, random_seed=KITCHEN_SEED)
        generator.generate().to_csv(path, index=False)
        return path

    return write_csv


# titanic and kitchen5k are the golden datasets, so a timing and a snapshot describe the
# same input. kitchen50k is the scaling case.
SCENARIOS = [
    Scenario("titanic", TITANIC.write_csv, repeats=3),
    Scenario("adversarial", _adversarial, repeats=3),
    Scenario(
        "kitchen5k", KITCHEN5K.write_csv, time_series_column=KITCHEN_TIME_SERIES_COLUMN
    ),
    Scenario(
        "kitchen50k",
        _kitchen(50_000, "kitchen50k"),
        time_series_column=KITCHEN_TIME_SERIES_COLUMN,
    ),
]


WIDE_ROWS = 1_000_000
WIDE_COLUMNS = 50
WIDE_SEED = 42


def _wide(directory: Path) -> Path:
    """A million rows of fifty float columns, as Parquet to be quick to read."""
    path = directory / "wide1m.parquet"
    rng = np.random.default_rng(WIDE_SEED)
    pl.DataFrame(
        {f"f{i:02d}": rng.standard_normal(WIDE_ROWS) for i in range(WIDE_COLUMNS)}
    ).write_parquet(path)
    return path


# The profile pass alone, on the scaling case and on the 1M x 50 case in the design.
PROFILE_SCENARIOS = [
    ProfileScenario("profile-kitchen50k", _kitchen(50_000, "kitchen50k")),
    ProfileScenario("profile-wide1m", _wide),
]


# The full legacy task set through ``dsbf.profile(depth="deep")``, serial and parallel,
# then the default selection at the standard depth, which the Phase 4 budgets in
# ``compare.ENGINE_BUDGETS`` are about.
ENGINE_SCENARIOS = [
    EngineScenario("engine-titanic-serial", TITANIC.write_csv, workers=1, repeats=3),
    EngineScenario(
        "engine-titanic-parallel", TITANIC.write_csv, workers=PER_CORE, repeats=3
    ),
    EngineScenario(
        "engine-kitchen5k-serial",
        KITCHEN5K.write_csv,
        workers=1,
        time_series_column=KITCHEN_TIME_SERIES_COLUMN,
    ),
    EngineScenario(
        "engine-kitchen5k-parallel",
        KITCHEN5K.write_csv,
        workers=PER_CORE,
        time_series_column=KITCHEN_TIME_SERIES_COLUMN,
    ),
    EngineScenario(
        "standard-titanic",
        TITANIC.write_csv,
        workers=None,
        repeats=3,
        depth="standard",
    ),
    EngineScenario(
        "standard-kitchen50k",
        _kitchen(50_000, "kitchen50k"),
        workers=None,
        time_series_column=KITCHEN_TIME_SERIES_COLUMN,
        depth="standard",
    ),
]
