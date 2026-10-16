"""The datasets with a committed golden snapshot, and how to materialise each."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tests.generators.dsbf_test_dataset_generator import (
    GENERATORS,
    DSBFTestDataGenerator,
)

GOLDEN_DIR = Path(__file__).parent
DATA_DIR = GOLDEN_DIR / "data"
SNAPSHOT_DIR = GOLDEN_DIR / "snapshots"

KITCHEN_ROWS = 5_000
KITCHEN_SEED = 42


@dataclass(frozen=True)
class GoldenDataset:
    """One dataset: a name, a writer for its CSV, and optional time-series settings."""

    name: str
    write_csv: Callable[[Path], Path]
    time_series_column: str | None = None

    @property
    def snapshot_path(self) -> Path:
        return SNAPSHOT_DIR / f"{self.name}.json"


def _archetype(name: str) -> GoldenDataset:
    def write_csv(directory: Path) -> Path:
        path = directory / f"{name}.csv"
        GENERATORS[name]().to_csv(path, index=False)
        return path

    return GoldenDataset(name, write_csv)


def _titanic(_directory: Path) -> Path:
    return DATA_DIR / "titanic.csv"


def _kitchen5k(directory: Path) -> Path:
    path = directory / "kitchen5k.csv"
    generator = DSBFTestDataGenerator(n_rows=KITCHEN_ROWS, random_seed=KITCHEN_SEED)
    generator.generate().to_csv(path, index=False)
    return path


# `comprehensive` is the kitchen-sink frame at its default size; kitchen5k is that same
# generator pinned to a seed, so it is not listed twice.
ARCHETYPES = [_archetype(name) for name in GENERATORS if name != "comprehensive"]
TITANIC = GoldenDataset("titanic", _titanic)
KITCHEN5K = GoldenDataset("kitchen5k", _kitchen5k, time_series_column="normal_date")

GOLDEN_DATASETS = [*ARCHETYPES, TITANIC, KITCHEN5K]
