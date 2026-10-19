"""The new inference agrees with the legacy ``infer_types`` task, except where listed.

Small datasets are compared with the committed golden snapshots. ``wide`` and
``kitchen5k`` have too many columns for the snapshot to hold the types, so the legacy
task runs on them live.
"""

import json

import pandas as pd
import polars as pl
import pytest

from dsbf.data import load
from dsbf.eda.tasks.infer_types import InferTypes
from dsbf.profiling import infer_column_types
from tests.golden.datasets import GOLDEN_DATASETS, GoldenDataset
from tests.test_profile.accepted import cells

# The legacy name for each type that was renamed ("semantic-type-names").
RENAMED = {"continuous": "numeric", "id": "identifier"}

# Differences by dataset and column: (legacy, new). A Boolean column is not listed one
# by one; see ``_boolean_difference``. The reasons are in ``accepted``.
ACCEPTED = {
    cell: ("identifier", "datetime") for cell in cells("datetime-past-pandas-range")
}


def _boolean_difference(legacy: str, new: str, dtype: pl.DataType) -> bool:
    """A bool column is boolean now; the legacy task calls it categorical."""
    return dtype == pl.Boolean and (legacy, new) == ("categorical", "boolean")


def _compare(dataset: GoldenDataset, path, legacy: dict[str, str]) -> None:
    loaded = load(path)
    types = infer_column_types(loaded)
    assert set(types) == set(legacy)
    unexplained = []
    listed = set()
    for column, old in legacy.items():
        old = RENAMED.get(old, old)
        new = types[column].semantic_type
        if old == new or _boolean_difference(old, new, loaded.schema[column]):
            continue
        accepted = ACCEPTED.get((dataset.name, column))
        if accepted is None or accepted != (old, new):
            unexplained.append((column, old, new, types[column].reason))
        else:
            listed.add(column)
    assert not unexplained, f"{dataset.name}: unlisted differences {unexplained}"
    # A listed difference that no longer happens should come off the list.
    expected = {c for (name, c) in ACCEPTED if name == dataset.name}
    assert listed == expected


def _live_legacy(path) -> dict[str, str]:
    task = InferTypes()
    task.set_input(pd.read_csv(path))
    task.run()
    assert task.output is not None
    assert task.output.data is not None
    return {
        column: info["analysis_intent_dtype"]
        for column, info in task.output.data.items()
    }


SNAPSHOTTED = [d for d in GOLDEN_DATASETS if d.name not in ("wide", "kitchen5k")]


@pytest.mark.unit
@pytest.mark.parametrize("dataset", SNAPSHOTTED, ids=lambda d: d.name)
def test_matches_the_golden_snapshot(dataset: GoldenDataset, tmp_path):
    snapshot = json.loads(dataset.snapshot_path.read_text())
    data = snapshot["tasks"]["infer_types"]["data"]
    legacy = {column: info["analysis_intent_dtype"] for column, info in data.items()}
    _compare(dataset, dataset.write_csv(tmp_path), legacy)


@pytest.mark.integration
@pytest.mark.parametrize(
    "dataset",
    [d for d in GOLDEN_DATASETS if d.name in ("wide", "kitchen5k")],
    ids=lambda d: d.name,
)
def test_matches_the_legacy_task_run_live(dataset: GoldenDataset, tmp_path):
    path = dataset.write_csv(tmp_path)
    _compare(dataset, path, _live_legacy(path))
