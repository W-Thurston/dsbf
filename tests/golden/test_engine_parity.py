"""Run each golden dataset through the new engine and compare it with its snapshot.

Run with ``uv run pytest -m golden``. The snapshots were made by the legacy engine, so
the two are expected to differ in places; ``engine_differences.json`` lists each
difference and the reason for it, and a run that differs in any other way fails. After a
change that moves a difference, review it and rewrite the file with ``--update-golden``;
see ``tests/golden/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.golden.datasets import GOLDEN_DATASETS, KITCHEN5K, GoldenDataset
from tests.golden.engine_compare import differences
from tests.golden.runner import profile_with_engine

DIFFERENCES = Path(__file__).parent / "engine_differences.json"


def _load() -> dict[str, Any]:
    return json.loads(DIFFERENCES.read_text())


def _param(dataset: GoldenDataset):
    marks = [pytest.mark.slow] if dataset is KITCHEN5K else []
    return pytest.param(dataset, id=dataset.name, marks=marks)


@pytest.mark.golden
@pytest.mark.parametrize("dataset", [_param(d) for d in GOLDEN_DATASETS])
def test_engine_run_matches_snapshot_but_for_the_listed_differences(
    dataset: GoldenDataset, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    run = profile_with_engine(dataset, tmp_path)
    snapshot = json.loads(dataset.snapshot_path.read_text())
    actual = differences(snapshot, run.report, run.roots)

    listed = _load()
    if request.config.getoption("--update-golden"):
        listed["differences"][dataset.name] = dict(sorted(actual.items()))
        DIFFERENCES.write_text(json.dumps(listed, indent=1, ensure_ascii=False) + "\n")
        return

    expected = listed["differences"].get(dataset.name, {})
    new = {t: d for t, d in actual.items() if expected.get(t) != d}
    gone = {t: d for t, d in expected.items() if actual.get(t) != d}
    if new or gone:
        lines = [f"+ {task}: {what}" for task, what in sorted(new.items())]
        lines += [f"- {task}: {what}" for task, what in sorted(gone.items())]
        pytest.fail(
            f"{dataset.name} through the engine differs from engine_differences.json. "
            "If the change is intended, review it and rerun with --update-golden.\n"
            + "\n".join(lines),
            pytrace=False,
        )


@pytest.mark.unit
def test_every_listed_difference_has_a_reason() -> None:
    listed = _load()

    for dataset, found in listed["differences"].items():
        for task, what in found.items():
            if what.startswith("changed"):
                reason = listed["changed"].get(task)
                assert reason in listed["reasons"], f"{dataset}: {task} has no reason"
            else:
                assert what.startswith("status:"), f"{dataset}: {task}: {what}"
    assert set(listed["changed"].values()) <= set(listed["reasons"])


@pytest.mark.unit
def test_every_golden_dataset_is_listed() -> None:
    assert set(_load()["differences"]) == {d.name for d in GOLDEN_DATASETS}
