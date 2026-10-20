"""The core pack's findings on each golden dataset match their committed snapshot.

Run with ``uv run pytest -m golden``. Review a difference, then accept it with
``--update-golden``; see ``tests/golden/README.md``.
"""

from __future__ import annotations

import difflib

import pytest

from tests.golden.datasets import GOLDEN_DATASETS, KITCHEN5K, GoldenDataset
from tests.golden.migration.snapshot import core_snapshot
from tests.golden.normalise import render_snapshot

pytestmark = pytest.mark.golden

MAX_DIFF_LINES = 120


def _param(dataset: GoldenDataset):
    marks = [pytest.mark.slow] if dataset is KITCHEN5K else []
    return pytest.param(dataset, id=dataset.name, marks=marks)


@pytest.mark.parametrize("dataset", [_param(d) for d in GOLDEN_DATASETS])
def test_core_findings_match_the_snapshot(
    dataset: GoldenDataset, engine_run, request: pytest.FixtureRequest
) -> None:
    run = engine_run(dataset)
    assert run.core is not None
    actual = render_snapshot(core_snapshot(dataset.name, run.core))
    path = dataset.core_snapshot_path

    if request.config.getoption("--update-golden"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual)
        return

    assert path.exists(), f"No core snapshot for {dataset.name}; use --update-golden."
    expected = path.read_text()
    if actual != expected:
        diff = list(
            difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=f"snapshots/core/{dataset.name}.json (committed)",
                tofile="this run",
                lineterm="",
                n=1,
            )
        )
        shown = "\n".join(diff[:MAX_DIFF_LINES])
        more = len(diff) - MAX_DIFF_LINES
        suffix = f"\n... {more} more diff lines" if more > 0 else ""
        pytest.fail(
            f"The core findings on {dataset.name} no longer match their snapshot. "
            "If the change is intended, review it and rerun with --update-golden.\n"
            + shown
            + suffix,
            pytrace=False,
        )
