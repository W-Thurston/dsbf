"""Compare a full run on each golden dataset with its committed snapshot.

Run with ``uv run pytest -m golden``. Review a difference, then accept it with
``--update-golden``; see ``tests/golden/README.md``.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

from tests.golden.datasets import GOLDEN_DATASETS, KITCHEN5K, GoldenDataset
from tests.golden.normalise import build_snapshot, render_snapshot
from tests.golden.runner import profile

pytestmark = pytest.mark.golden

MAX_DIFF_LINES = 120


def _param(dataset: GoldenDataset):
    marks = [pytest.mark.slow] if dataset is KITCHEN5K else []
    return pytest.param(dataset, id=dataset.name, marks=marks)


@pytest.mark.parametrize("dataset", [_param(d) for d in GOLDEN_DATASETS])
def test_run_matches_snapshot(
    dataset: GoldenDataset, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    run = profile(dataset, tmp_path)
    actual = render_snapshot(
        build_snapshot(dataset.name, run.report, run.run_info, run.roots)
    )

    if request.config.getoption("--update-golden"):
        dataset.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.snapshot_path.write_text(actual)
        return

    assert dataset.snapshot_path.exists(), (
        f"No snapshot for {dataset.name}; create it with --update-golden."
    )
    expected = dataset.snapshot_path.read_text()
    if actual != expected:
        diff = list(
            difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=f"snapshots/{dataset.name}.json (committed)",
                tofile="this run",
                lineterm="",
                n=1,
            )
        )
        shown = "\n".join(diff[:MAX_DIFF_LINES])
        more = len(diff) - MAX_DIFF_LINES
        suffix = f"\n... {more} more diff lines" if more > 0 else ""
        pytest.fail(
            f"{dataset.name} no longer matches its golden snapshot. If the change is "
            "intended, review it and rerun with --update-golden.\n" + shown + suffix,
            pytrace=False,
        )
