"""The datasets with a committed golden snapshot, and how to materialise each."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    core_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Parameters for the core checks that have no default to run on this dataset."""

    @property
    def snapshot_path(self) -> Path:
        return SNAPSHOT_DIR / f"{self.name}.json"

    @property
    def core_snapshot_path(self) -> Path:
        """The core pack's findings on this dataset."""
        return SNAPSHOT_DIR / "core" / f"{self.name}.json"


def _archetype(name: str) -> GoldenDataset:
    def write_csv(directory: Path) -> Path:
        path = directory / f"{name}.csv"
        GENERATORS[name]().to_csv(path, index=False)
        return path

    return GoldenDataset(name, write_csv)


def _titanic(_directory: Path) -> Path:
    return DATA_DIR / "titanic.csv"


def _sms_spam(directory: Path) -> Path:
    """The vendored tab-separated corpus, with a header row, as a CSV."""
    path = directory / "sms_spam.csv"
    lines = (DATA_DIR / "SMSSpamCollection").read_text(encoding="utf-8").split("\n")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "text"])
        writer.writerows(line.split("\t", 1) for line in lines if line)
    return path


def _kitchen5k(directory: Path) -> Path:
    path = directory / "kitchen5k.csv"
    generator = DSBFTestDataGenerator(n_rows=KITCHEN_ROWS, random_seed=KITCHEN_SEED)
    generator.generate().to_csv(path, index=False)
    return path


# `comprehensive` is the kitchen-sink frame at its default size; kitchen5k is that same
# generator pinned to a seed, so it is not listed twice.
ARCHETYPES = [_archetype(name) for name in GENERATORS if name != "comprehensive"]
TITANIC = GoldenDataset("titanic", _titanic)
# The generator plants a target and the columns that leak it; no other dataset has one.
KITCHEN5K = GoldenDataset(
    "kitchen5k",
    _kitchen5k,
    time_series_column="normal_date",
    core_params={
        "target_leakage": {"target": "target_for_leakage"},
        "mutual_information": {"target": "target_for_leakage"},
    },
)
# A message is plain text: it names no link, and a label is one of the two classes.
SMS_SPAM = GoldenDataset(
    "sms_spam",
    _sms_spam,
    core_params={
        "format_violations": {
            "rules": [
                {"column": "label", "pattern": "ham|spam"},
                {"column": "text", "pattern": r"(?!.*(?:https?://|www\.)).*"},
            ]
        }
    },
)

GOLDEN_DATASETS = [*ARCHETYPES, TITANIC, KITCHEN5K, SMS_SPAM]
