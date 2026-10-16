"""The datasets that are timed, and how to materialise each."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tests.fixtures.adversarial import build_adversarial_frame
from tests.generators.dsbf_test_dataset_generator import DSBFTestDataGenerator
from tests.golden.datasets import KITCHEN5K, TITANIC

KITCHEN_SEED = 42
KITCHEN_TIME_SERIES_COLUMN = "normal_date"


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
